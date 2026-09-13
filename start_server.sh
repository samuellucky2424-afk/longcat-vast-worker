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
export PYTHONPATH="${LONGCAT_REPO_DIR}:${PYTHONPATH:-}"

mkdir -p /workspace/models /workspace/hf-cache /var/log/longcat

# Install bootstrap/runtime packages unconditionally. The CUDA base image is minimal
# and may not include python, pip, git, curl or ffmpeg.
apt-get update
apt-get install -y --no-install-recommends \
  python3.10 python3.10-dev python3-pip python-is-python3 \
  git curl ca-certificates ffmpeg build-essential ninja-build \
  libgl1 libglib2.0-0
rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip setuptools wheel packaging ninja psutil

if ! python - <<'PY' >/dev/null 2>&1
import torch
assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
PY
then
  python -m pip install \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
fi

if [ ! -d "${PYWORKER_DIR}/.git" ]; then
  rm -rf "${PYWORKER_DIR}"
  git clone --depth 1 "${PYWORKER_REPO:-https://github.com/samuellucky2424-afk/longcat-vast-worker.git}" "${PYWORKER_DIR}"
else
  git -C "${PYWORKER_DIR}" pull --ff-only || true
fi

python -m pip install -r "${PYWORKER_DIR}/requirements.txt"

if [ ! -d "${LONGCAT_REPO_DIR}/.git" ]; then
  rm -rf "${LONGCAT_REPO_DIR}"
  git clone --depth 1 --branch main https://github.com/meituan-longcat/LongCat-Video.git "${LONGCAT_REPO_DIR}"
else
  git -C "${LONGCAT_REPO_DIR}" pull --ff-only || true
fi

grep -vE '^(torch|torchvision|torchaudio|flash-attn)==' "${LONGCAT_REPO_DIR}/requirements.txt" > /tmp/longcat-requirements.txt
python -m pip install -r /tmp/longcat-requirements.txt

if ! python - <<'PY' >/dev/null 2>&1
import flash_attn
PY
then
  MAX_JOBS=8 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

python -m pip install 'huggingface_hub>=0.36,<1' hf_xet

if [ ! -f "${MODEL_DIR}/model_index.json" ] && [ ! -d "${MODEL_DIR}/dit" ]; then
  echo "LONGCAT_LOADING downloading ${MODEL_NAME} to ${MODEL_DIR}"
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_NAME"],
    local_dir=os.environ["MODEL_DIR"],
)
PY
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${PYWORKER_DIR}"

python -u model_server.py > /var/log/longcat/model.log 2>&1 &
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

exec python -u worker.py
