import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import torch
import torch.distributed as dist
from botocore.config import Config
from flask import Flask, jsonify, request
from PIL import Image
from torchvision.io import read_video, write_video
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


MODEL_ID = os.environ.get("MODEL_NAME", "meituan-longcat/LongCat-Video")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models/LongCat-Video"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/longcat_outputs"))
PORT = int(os.environ.get("MODEL_SERVER_PORT", "18000"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "").rstrip("/")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "outputs/longcat").strip("/")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
_S3_CLIENT = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("AWS_RETRY_MODE", "standard")
os.environ.setdefault("AWS_MAX_ATTEMPTS", "10")

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def log(message: str):
    print(message, flush=True)


def storage_configured() -> bool:
    return all([
        S3_ENDPOINT_URL,
        S3_BUCKET,
        S3_REGION,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
    ])


def get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    if not storage_configured():
        raise RuntimeError(
            "S3 output storage is not configured. Set S3_ENDPOINT_URL, S3_BUCKET, "
            "S3_REGION, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY."
        )
    _S3_CLIENT = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
        endpoint_url=S3_ENDPOINT_URL,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=10,
            read_timeout=300,
            s3={"addressing_style": "path"},
        ),
    )
    return _S3_CLIENT


def build_storage_key(filename: str) -> str:
    now = datetime.now(timezone.utc)
    dated_prefix = f"{now:%Y/%m/%d}"
    return f"{S3_PREFIX}/{dated_prefix}/{filename}" if S3_PREFIX else f"{dated_prefix}/{filename}"


def validate_storage_key(key: str) -> str:
    key = str(key or "").strip().lstrip("/")
    if not key:
        raise ValueError("source_key is required.")
    if ".." in key.split("/"):
        raise ValueError("source_key contains an invalid path component.")
    if S3_PREFIX and not key.startswith(f"{S3_PREFIX}/"):
        raise ValueError(f"source_key must point inside '{S3_PREFIX}/'.")
    return key


def upload_video(local_path: Path) -> dict:
    key = build_storage_key(local_path.name)
    get_s3_client().upload_file(
        str(local_path),
        S3_BUCKET,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    return {
        "provider": "s3_compatible",
        "bucket": S3_BUCKET,
        "key": key,
        "s3_uri": f"s3://{S3_BUCKET}/{key}",
        "endpoint_url": S3_ENDPOINT_URL,
    }


def download_video(key: str) -> Path:
    key = validate_storage_key(key)
    local_path = OUTPUT_DIR / f"source_{uuid.uuid4().hex}.mp4"
    get_s3_client().download_file(S3_BUCKET, key, str(local_path))
    if not local_path.exists() or local_path.stat().st_size == 0:
        local_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded source video is empty: {key}")
    return local_path


def init_distributed() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("LongCat-Video requires an NVIDIA CUDA GPU.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
    init_context_parallel(
        context_parallel_size=1,
        global_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )
    return local_rank


def load_pipeline():
    if not MODEL_DIR.is_dir():
        raise RuntimeError(
            f"LongCat model directory is missing: {MODEL_DIR}. "
            "start_server.sh must download the model before launching this process."
        )

    local_rank = init_distributed()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    log(f"LONGCAT_LOADING model={MODEL_ID} path={MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), subfolder="tokenizer", local_files_only=True
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(MODEL_DIR),
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        str(MODEL_DIR),
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(MODEL_DIR), subfolder="scheduler", local_files_only=True
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        str(MODEL_DIR),
        subfolder="dit",
        cp_split_hw=cp_split_hw,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    lora_path = MODEL_DIR / "lora" / "cfg_step_lora.safetensors"
    pipe.dit.load_lora(str(lora_path), "cfg_step_lora")
    pipe.dit.enable_loras(["cfg_step_lora"])
    log(f"LONGCAT_READY gpu={torch.cuda.get_device_name(local_rank)}")
    return pipe, local_rank


PIPE, LOCAL_RANK = load_pipeline()


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def require_int(value, minimum: int, maximum: int, name: str) -> int:
    value = int(value)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def save_and_upload_frames(frames, fps: int, crf: int = 28):
    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"
    video_tensor = torch.from_numpy(np.asarray(frames))
    if video_tensor.dtype != torch.uint8:
        video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8)
    try:
        write_video(
            str(output_path),
            video_tensor,
            fps=fps,
            video_codec="libx264",
            options={"crf": str(crf)},
        )
        size_bytes = output_path.stat().st_size
        storage = upload_video(output_path)
    finally:
        output_path.unlink(missing_ok=True)
    return size_bytes, storage


def load_source_frames(source_path: Path, target_fps: int):
    tensor, _, info = read_video(str(source_path), pts_unit="sec")
    if tensor.ndim != 4 or tensor.shape[0] == 0:
        raise ValueError("Source video contains no readable frames.")
    source_fps = float(info.get("video_fps") or target_fps)
    stride = max(1, round(source_fps / target_fps))
    sampled = tensor[::stride]
    frames = [Image.fromarray(frame.numpy()).convert("RGB") for frame in sampled]
    return frames, source_fps, stride


def generate_text_to_video(job: dict):
    if not storage_configured():
        raise RuntimeError("Output storage is not configured.")

    prompt = str(job.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required.")

    width = int(job.get("width", 832))
    height = int(job.get("height", 480))
    if (width, height) != (832, 480):
        raise ValueError("This worker currently supports 832x480 only.")

    num_frames = require_int(job.get("num_frames", 49), 17, 93, "num_frames")
    if num_frames % 4 != 1:
        raise ValueError("num_frames must follow 4n+1, for example 49 or 93.")
    steps = require_int(job.get("steps", 16), 8, 50, "steps")
    fps = require_int(job.get("fps", 15), 8, 30, "fps")
    seed = int(job.get("seed", 42))
    use_distill = bool(job.get("use_distill", True))

    log(f"LONGCAT_GENERATING action=text_to_video frames={num_frames} steps={steps}")
    generator = torch.Generator(device=LOCAL_RANK).manual_seed(seed)

    if use_distill:
        PIPE.dit.enable_loras(["cfg_step_lora"])
        frames = PIPE.generate_t2v(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            use_distill=True,
            guidance_scale=1.0,
            generator=generator,
        )[0]
    else:
        PIPE.dit.disable_all_loras()
        frames = PIPE.generate_t2v(
            prompt=prompt,
            negative_prompt=str(job.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=float(job.get("guidance_scale", 4.0)),
            generator=generator,
        )[0]
        PIPE.dit.enable_loras(["cfg_step_lora"])

    size_bytes, storage = save_and_upload_frames(frames, fps, int(job.get("crf", 28)))
    result = {
        "model": MODEL_ID,
        "action": "text_to_video",
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": round(num_frames / fps, 3),
        "seed": seed,
        "use_distill": use_distill,
        "size_bytes": size_bytes,
        "mime_type": "video/mp4",
        "storage": storage,
    }
    del frames
    torch_gc()
    return result


def generate_video_continuation(job: dict):
    if not storage_configured():
        raise RuntimeError("Output storage is not configured.")

    source_key = validate_storage_key(job.get("source_key"))
    prompt = str(job.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required.")

    fps = require_int(job.get("fps", 15), 8, 30, "fps")
    num_frames = require_int(job.get("num_frames", 93), 49, 93, "num_frames")
    if num_frames % 4 != 1:
        raise ValueError("num_frames must follow 4n+1, for example 49 or 93.")
    num_cond_frames = int(job.get("num_cond_frames", 13))
    if num_cond_frames != 13:
        raise ValueError("num_cond_frames must currently be 13.")
    steps = require_int(job.get("steps", 16), 8, 50, "steps")
    seed = int(job.get("seed", 42))
    use_distill = bool(job.get("use_distill", True))

    source_path = download_video(source_key)
    try:
        source_frames, source_fps, stride = load_source_frames(source_path, fps)
    finally:
        source_path.unlink(missing_ok=True)

    if len(source_frames) < num_cond_frames:
        raise ValueError(f"Source has only {len(source_frames)} sampled frames.")
    target_size = source_frames[0].size
    if target_size != (832, 480):
        raise ValueError(f"Continuation expects 832x480, got {target_size[0]}x{target_size[1]}.")

    log(f"LONGCAT_GENERATING action=continue_video frames={num_frames} steps={steps}")
    generator = torch.Generator(device=LOCAL_RANK).manual_seed(seed)

    if use_distill:
        PIPE.dit.enable_loras(["cfg_step_lora"])
        generated = PIPE.generate_vc(
            video=source_frames,
            prompt=prompt,
            resolution="480p",
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=steps,
            use_distill=True,
            guidance_scale=1.0,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=False,
        )[0]
    else:
        PIPE.dit.disable_all_loras()
        generated = PIPE.generate_vc(
            video=source_frames,
            prompt=prompt,
            negative_prompt=str(job.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
            resolution="480p",
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=steps,
            guidance_scale=float(job.get("guidance_scale", 4.0)),
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
        )[0]
        PIPE.dit.enable_loras(["cfg_step_lora"])

    new_frames = [
        Image.fromarray((generated[i] * 255).clip(0, 255).astype(np.uint8)).resize(target_size, Image.BICUBIC)
        for i in range(generated.shape[0])
    ]
    combined_frames = source_frames + new_frames[num_cond_frames:]
    combined_np = np.stack([np.asarray(frame, dtype=np.uint8) for frame in combined_frames])
    size_bytes, storage = save_and_upload_frames(combined_np, fps, int(job.get("crf", 28)))

    result = {
        "model": MODEL_ID,
        "action": "continue_video",
        "source": {
            "bucket": S3_BUCKET,
            "key": source_key,
            "source_fps": source_fps,
            "sampling_stride": stride,
            "sampled_frames": len(source_frames),
        },
        "width": target_size[0],
        "height": target_size[1],
        "fps": fps,
        "conditioning_frames": num_cond_frames,
        "generated_segment_frames": num_frames,
        "new_frames_added": len(new_frames) - num_cond_frames,
        "total_frames": len(combined_frames),
        "duration_seconds": round(len(combined_frames) / fps, 3),
        "seed": seed,
        "use_distill": use_distill,
        "size_bytes": size_bytes,
        "mime_type": "video/mp4",
        "storage": storage,
    }
    del generated, new_frames, combined_frames, combined_np, source_frames
    torch_gc()
    return result


def segment_prompt(job: dict, index: int, fallback: str) -> str:
    prompts = job.get("prompts")
    if isinstance(prompts, list) and prompts:
        if index < len(prompts) and str(prompts[index]).strip():
            return str(prompts[index]).strip()
        if str(prompts[-1]).strip():
            return str(prompts[-1]).strip()
    return fallback


def generate_continuation_chain(job: dict):
    if not storage_configured():
        raise RuntimeError("Output storage is not configured.")

    source_key = validate_storage_key(job.get("source_key"))
    prompt = str(job.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required.")
    fps = require_int(job.get("fps", 15), 8, 30, "fps")
    num_frames = require_int(job.get("num_frames", 93), 49, 93, "num_frames")
    if num_frames % 4 != 1:
        raise ValueError("num_frames must follow 4n+1.")
    num_cond_frames = int(job.get("num_cond_frames", 13))
    if num_cond_frames != 13:
        raise ValueError("num_cond_frames must currently be 13.")
    segments = require_int(job.get("continuation_segments", 2), 1, 11, "continuation_segments")
    steps = require_int(job.get("steps", 16), 8, 50, "steps")
    seed = int(job.get("seed", 42))
    use_distill = bool(job.get("use_distill", True))
    crf = int(job.get("crf", 28))

    source_path = download_video(source_key)
    try:
        source_frames, source_fps, stride = load_source_frames(source_path, fps)
    finally:
        source_path.unlink(missing_ok=True)

    if len(source_frames) < num_cond_frames:
        raise ValueError(f"Source has only {len(source_frames)} sampled frames.")
    target_size = source_frames[0].size
    if target_size != (832, 480):
        raise ValueError(f"continue_chain expects 832x480, got {target_size[0]}x{target_size[1]}.")

    generator = torch.Generator(device=LOCAL_RANK).manual_seed(seed)
    current_video = source_frames[-num_frames:] if len(source_frames) >= num_frames else source_frames
    all_frames = list(source_frames)
    segment_stats = []

    for index in range(segments):
        current_prompt = segment_prompt(job, index, prompt)
        log(f"LONGCAT_GENERATING action=continue_chain segment={index + 1}/{segments}")
        if use_distill:
            PIPE.dit.enable_loras(["cfg_step_lora"])
            generated = PIPE.generate_vc(
                video=current_video,
                prompt=current_prompt,
                resolution="480p",
                num_frames=num_frames,
                num_cond_frames=num_cond_frames,
                num_inference_steps=steps,
                use_distill=True,
                guidance_scale=1.0,
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=False,
                enhance_hf=False,
            )[0]
        else:
            PIPE.dit.disable_all_loras()
            generated = PIPE.generate_vc(
                video=current_video,
                prompt=current_prompt,
                negative_prompt=str(job.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
                resolution="480p",
                num_frames=num_frames,
                num_cond_frames=num_cond_frames,
                num_inference_steps=steps,
                guidance_scale=float(job.get("guidance_scale", 4.0)),
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=False,
                enhance_hf=True,
            )[0]
            PIPE.dit.enable_loras(["cfg_step_lora"])

        new_video = [
            Image.fromarray((generated[i] * 255).clip(0, 255).astype(np.uint8)).resize(target_size, Image.BICUBIC)
            for i in range(generated.shape[0])
        ]
        added = new_video[num_cond_frames:]
        all_frames.extend(added)
        current_video = new_video
        segment_stats.append({
            "segment": index + 1,
            "prompt": current_prompt,
            "generated_frames": len(new_video),
            "conditioning_frames": num_cond_frames,
            "new_frames_added": len(added),
            "cumulative_frames": len(all_frames),
            "cumulative_duration_seconds": round(len(all_frames) / fps, 3),
        })
        del generated, added
        torch_gc()

    combined_np = np.stack([np.asarray(frame, dtype=np.uint8) for frame in all_frames])
    size_bytes, storage = save_and_upload_frames(combined_np, fps, crf)
    result = {
        "model": MODEL_ID,
        "action": "continue_chain",
        "source": {
            "bucket": S3_BUCKET,
            "key": source_key,
            "source_fps": source_fps,
            "sampling_stride": stride,
            "source_frames": len(source_frames),
        },
        "width": target_size[0],
        "height": target_size[1],
        "fps": fps,
        "continuation_segments": segments,
        "conditioning_frames_per_segment": num_cond_frames,
        "generated_segment_frames": num_frames,
        "new_frames_added_per_segment": num_frames - num_cond_frames,
        "total_new_frames_added": segments * (num_frames - num_cond_frames),
        "total_frames": len(all_frames),
        "duration_seconds": round(len(all_frames) / fps, 3),
        "seed": seed,
        "use_distill": use_distill,
        "segments": segment_stats,
        "size_bytes": size_bytes,
        "mime_type": "video/mp4",
        "storage": storage,
    }
    del combined_np, all_frames, current_video, source_frames
    torch_gc()
    return result


def unwrap_payload(payload: dict) -> dict:
    if isinstance(payload.get("input"), dict):
        return payload["input"]
    return payload


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(LOCAL_RANK),
        "storage_configured": storage_configured(),
    })


@app.post("/generate")
def generate():
    try:
        payload = request.get_json(silent=False) or {}
        job = unwrap_payload(payload)
        action = str(job.get("action", "text_to_video"))

        if action == "health":
            result = {
                "ok": True,
                "model": MODEL_ID,
                "gpu": torch.cuda.get_device_name(LOCAL_RANK),
                "storage_configured": storage_configured(),
            }
        elif action == "storage_health":
            if not storage_configured():
                raise RuntimeError("S3 storage is not configured.")
            get_s3_client().list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX, MaxKeys=1)
            result = {"ok": True, "bucket": S3_BUCKET, "prefix": S3_PREFIX}
        elif action == "text_to_video":
            result = generate_text_to_video(job)
        elif action in {"continue_video", "video_continuation"}:
            result = generate_video_continuation(job)
        elif action in {"continue_chain", "long_continue"}:
            result = generate_continuation_chain(job)
        else:
            raise ValueError(f"Unsupported action: {action}")

        return jsonify(result), 200
    except ValueError as exc:
        torch_gc()
        return jsonify({"error": type(exc).__name__, "message": str(exc)}), 400
    except Exception as exc:
        log(f"LONGCAT_FATAL {type(exc).__name__}: {exc}")
        torch_gc()
        return jsonify({"error": type(exc).__name__, "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, threaded=False)
