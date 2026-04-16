"""Prometheus metric definitions for the inference API."""
from prometheus_client import Counter, Histogram, Gauge, Info

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_COUNTER = Counter(
    "prediction_total",
    "Total number of predictions served",
    labelnames=["label"],  # "churn" | "no_churn"
)

DRIFT_SCORE = Gauge(
    "drift_score_current",
    "Most recent Evidently drift score (0.0 – 1.0)",
)

DATASET_DRIFT_FLAG = Gauge(
    "dataset_drift_detected",
    "1 if dataset-level drift was detected in the latest check, else 0",
)

MODEL_INFO = Info(
    "model",
    "Currently loaded model metadata",
)


def update_model_info(run_id: str, stage: str, model_name: str) -> None:
    MODEL_INFO.info({
        "run_id": run_id,
        "stage": stage,
        "model_name": model_name,
    })
