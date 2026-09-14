#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_NAME="${MODEL_NAME:-meituan-longcat/LongCat-Video}"
export MODEL_DIR="${MODEL_DIR:-/workspace/models/LongCat-Video}"
export MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-18000}"
export PYWORKER_DIR="${PYWORKER_DIR:-/workspace/longcat-vast-worker}"
export LONGCAT_REPO_DIR="${LONGCAT_REPO_DIR:-/workspace/LongCat-Video}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export MODEL_VENV_DIR="${MODEL_VENV_DIR:-/workspace/longcat-venv}"
export WORKER_VENV_DIR="${WORKER_VENV_DIR:-/workspace/vast-worker-venv}"
export PYTHONPATH="${LONGCAT_REPO_DIR}:${PYTHONPATH:-}"

mkdir -p /workspace/models /workspace/hf-cache /var/log/longcat

apt-get update
apt-get install -y --no-install-recommends \
  python3.10 python3.10-dev python3.10-venv python3-pip python-is-python3 \
  git curl ca-certificates ffmpeg build-essential ninja-build \
  libgl1 libglib2.0-0
rm -rf /var/lib/apt/lists/*

if [ ! -d "${PYWORKER_DIR}/.git" ]; then
  rm -rf "${PYWORKER_DIR}"
  git clone --depth 1 "${PYWORKER_REPO:-https://github.com/samuellucky2424-afk/longcat-vast-worker.git}" "${PYWORKER_DIR}"
else
  git -C "${PYWORKER_DIR}" pull --ff-only || true
fi

if [ ! -d "${LONGCAT_REPO_DIR}/.git" ]; then
  rm -rf "${LONGCAT_REPO_DIR}"
  git clone --depth 1 --branch main https://github.com/meituan-longcat/LongCat-Video.git "${LONGCAT_REPO_DIR}"
else
  git -C "${LONGCAT_REPO_DIR}" pull --ff-only || true
fi

# Model environment: LongCat and its pinned inference dependencies live here.
if [ ! -x "${MODEL_VENV_DIR}/bin/python" ]; then
  rm -rf "${MODEL_VENV_DIR}"
  python3.10 -m venv "${MODEL_VENV_DIR}"
fi
MODEL_PY="${MODEL_VENV_DIR}/bin/python"

"${MODEL_PY}" -m pip install --upgrade pip setuptools wheel packaging ninja

if ! "${MODEL_PY}" - <<'PY' >/dev/null 2>&1
import torch
assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
PY
then
  "${MODEL_PY}" -m pip install \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
fi

# Dependencies used directly by model_server.py.
"${MODEL_PY}" -m pip install \
  'flask>=3.0,<4' \
  'boto3>=1.34,<2' \
  'huggingface_hub>=0.36,<1' \
  hf_xet

# LongCat keeps its own pins (including psutil/Pillow-related dependencies) in the
# model environment so they cannot conflict with the Vast worker SDK.
grep -vE '^(torch|torchvision|torchaudio|flash-attn)==' \
  "${LONGCAT_REPO_DIR}/requirements.txt" > /tmp/longcat-requirements.txt
"${MODEL_PY}" -m pip install -r /tmp/longcat-requirements.txt

if ! "${MODEL_PY}" - <<'PY' >/dev/null 2>&1
import flash_attn
PY
then
  MAX_JOBS=8 "${MODEL_PY}" -m pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

# Always run snapshot_download. It resumes partial downloads and becomes a fast
# cache check once all LongCat files are present, preventing partial-model boots.
echo "LONGCAT_LOADING ensuring ${MODEL_NAME} is complete in ${MODEL_DIR}"
"${MODEL_PY}" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_NAME"],
    local_dir=os.environ["MODEL_DIR"],
)
PY

# Worker environment: keep Vast SDK isolated from LongCat's incompatible pins.
if [ ! -x "${WORKER_VENV_DIR}/bin/python" ]; then
  rm -rf "${WORKER_VENV_DIR}"
  python3.10 -m venv "${WORKER_VENV_DIR}"
fi
WORKER_PY="${WORKER_VENV_DIR}/bin/python"
"${WORKER_PY}" -m pip install --upgrade pip setuptools wheel
"${WORKER_PY}" -m pip install -r "${PYWORKER_DIR}/requirements.txt"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${PYWORKER_DIR}"
: > /var/log/longcat/model.log
"${MODEL_PY}" -u model_server.py > /var/log/longcat/model.log 2>&1 &
MODEL_PID=$!

for i in $(seq 1 1800); do
  if grep -q "LONGCAT_READY" /var/log/longcat/model.log 2>/dev/null; then
    break
  fi
  if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
    cat /var/log/longcat/model.log || true
    exit 1
  fi
  sleep 1
done

if ! grep -q "LONGCAT_READY" /var/log/longcat/model.log 2>/dev/null; then
  echo "LongCat did not become ready before timeout"
  cat /var/log/longcat/model.log || true
  exit 1
fi

exec "${WORKER_PY}" -u worker.py
