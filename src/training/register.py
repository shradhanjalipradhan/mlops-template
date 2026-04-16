"""MLflow model registration and stage transition logic."""
import mlflow
from mlflow.tracking import MlflowClient
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


def register_model(run_id: str, auc: float) -> None:
    """
    Transition the model from this run_id to Production.
    Archives any previously Production model.
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    model_name = settings.model_name

    # Get the registered model version from this run
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        log.error("No registered model version found for run", run_id=run_id)
        return

    new_version = versions[0].version

    # Archive any existing Production versions
    existing = client.get_latest_versions(model_name, stages=["Production"])
    for v in existing:
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Archived",
        )
        log.info("Archived previous Production model", version=v.version)

    # Promote new version
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Production",
    )
    client.update_model_version(
        name=model_name,
        version=new_version,
        description=f"Promoted to Production. val_roc_auc={auc:.4f}. run_id={run_id}",
    )

    log.info(
        "Model promoted to Production",
        model_name=model_name,
        version=new_version,
        val_roc_auc=auc,
        run_id=run_id,
    )
