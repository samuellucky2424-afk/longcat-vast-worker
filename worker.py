import os

from vastai import HandlerConfig, LogActionConfig, Worker, WorkerConfig


MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.getenv("MODEL_SERVER_PORT", "18000"))
MODEL_LOG_FILE = os.getenv("MODEL_LOG_FILE", "/var/log/longcat/model.log")


def video_workload(payload: dict) -> float:
    """Estimate relative video-generation cost for Vast autoscaling."""
    width = int(payload.get("width", 832))
    height = int(payload.get("height", 480))
    num_frames = int(payload.get("num_frames", 93))

    baseline = 832 * 480 * 93
    return max(1.0, (width * height * num_frames) / baseline)


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
        on_load=[
            "LONGCAT_READY",
        ],
        on_error=[
            "LONGCAT_FATAL",
            "Traceback (most recent call last):",
        ],
        on_info=[
            "LONGCAT_LOADING",
            "LONGCAT_GENERATING",
        ],
    ),
)


if __name__ == "__main__":
    Worker(worker_config).run()
