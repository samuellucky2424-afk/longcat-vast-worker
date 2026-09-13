import os

from vastai import HandlerConfig, LogActionConfig, Worker, WorkerConfig


MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.getenv("MODEL_SERVER_PORT", "18000"))
MODEL_LOG_FILE = os.getenv("MODEL_LOG_FILE", "/var/log/longcat/model.log")


def _job(payload: dict) -> dict:
    nested = payload.get("input")
    return nested if isinstance(nested, dict) else payload


def video_workload(payload: dict) -> float:
    """Estimate relative LongCat cost for Vast autoscaling."""
    job = _job(payload)
    width = int(job.get("width", 832))
    height = int(job.get("height", 480))
    num_frames = int(job.get("num_frames", 93))
    segments = int(job.get("continuation_segments", 1))

    baseline = 832 * 480 * 93
    return max(1.0, (width * height * num_frames * max(1, segments)) / baseline)


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    handlers=[
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=False,
            max_queue_time=1800.0,
            workload_calculator=video_workload,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["LONGCAT_READY"],
        on_error=[
            "LONGCAT_FATAL",
            "Traceback (most recent call last):",
            "CUDA out of memory",
        ],
        on_info=[
            "LONGCAT_LOADING",
            "LONGCAT_GENERATING",
        ],
    ),
)


if __name__ == "__main__":
    Worker(worker_config).run()
