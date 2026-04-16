"""Model loader and prediction logic."""
import mlflow
import mlflow.sklearn
import pandas as pd
from src.serving.schemas import PredictRequest
from src.features.pipeline import add_engineered_features
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class Predictor:
    def __init__(self, model, run_id: str, stage: str):
        self.model = model
        self.run_id = run_id
        self.stage = stage

    @classmethod
    def from_registry(
        cls,
        tracking_uri: str,
        model_name: str,
        stage: str = "Production",
    ) -> "Predictor":
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}/{stage}"
        log.info("Loading model from registry", uri=model_uri)

        model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        versions = client.get_latest_versions(model_name, stages=[stage])
        run_id = versions[0].run_id if versions else "unknown"

        log.info("Model loaded", run_id=run_id, stage=stage)
        return cls(model=model, run_id=run_id, stage=stage)

    def predict(self, request: PredictRequest) -> tuple[str, float]:
        """Return (label, probability) for a single request."""
        row = pd.DataFrame([{
            "age": request.age,
            "tenure_months": request.tenure_months,
            "monthly_charge": request.monthly_charge,
            "num_products": request.num_products,
            "contract_type": request.contract_type,
            "payment_method": request.payment_method,
        }])
        row = add_engineered_features(row)

        proba = float(self.model.predict_proba(row)[0, 1])
        label = "churn" if proba >= 0.5 else "no_churn"
        return label, proba
