"""FastAPI inference service for churn prediction."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.serving.schemas import PredictRequest, PredictResponse, HealthResponse
from src.serving.predictor import Predictor
from src.serving.logger import PredictionLogger
from src.serving.metrics import (
    INFERENCE_LATENCY,
    PREDICTION_COUNTER,
    update_model_info,
)
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

# Module-level singletons — initialised during lifespan startup
predictor: Predictor | None = None
pred_logger: PredictionLogger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, pred_logger

    log.info(
        "Starting inference service",
        model_name=settings.model_name,
        stage=settings.model_stage,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    predictor = Predictor.from_registry(
        tracking_uri=settings.mlflow_tracking_uri,
        model_name=settings.model_name,
        stage=settings.model_stage,
    )
    pred_logger = PredictionLogger(db_path=settings.db_path)
    update_model_info(
        run_id=predictor.run_id,
        stage=predictor.stage,
        model_name=settings.model_name,
    )
    log.info("Service ready", run_id=predictor.run_id)

    yield

    pred_logger.close()
    log.info("Service shutdown complete")


app = FastAPI(
    title="Churn Prediction API",
    description="Real-time inference API for customer churn prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness + readiness probe."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return HealthResponse(
        status="ok",
        model_run_id=predictor.run_id,
        model_stage=predictor.stage,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Run churn prediction for a single customer.

    Returns the predicted label (churn | no_churn), the churn probability,
    and the inference latency in milliseconds.
    """
    if predictor is None or pred_logger is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    t0 = time.perf_counter()
    label, probability = predictor.predict(request)
    latency = time.perf_counter() - t0

    # Instrument
    INFERENCE_LATENCY.observe(latency)
    PREDICTION_COUNTER.labels(label=label).inc()

    # Persist
    pred_logger.log(request=request, label=label, probability=probability)

    log.info(
        "Prediction served",
        label=label,
        churn_probability=f"{probability:.4f}",
        latency_ms=f"{latency * 1000:.2f}",
    )

    return PredictResponse(
        label=label,
        churn_probability=probability,
        latency_ms=latency * 1000,
    )


@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({"message": "Churn Prediction API — see /docs for usage."})
