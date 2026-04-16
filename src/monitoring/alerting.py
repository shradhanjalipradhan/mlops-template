"""Send drift alerts and trigger Airflow retraining DAG via REST API."""
import requests
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


def trigger_retrain_dag(dag_id: str = "retrain_dag") -> bool:
    """
    Call the Airflow REST API to trigger a DAG run.
    Returns True on success, False on failure.
    """
    url = f"{settings.airflow_base_url}/api/v1/dags/{dag_id}/dagRuns"
    try:
        response = requests.post(
            url,
            json={"conf": {"triggered_by": "drift_alert"}},
            auth=(settings.airflow_user, settings.airflow_password),
            timeout=10,
        )
        if response.status_code in (200, 201):
            log.info("Retraining DAG triggered", dag_id=dag_id, status=response.status_code)
            return True
        else:
            log.error(
                "Failed to trigger DAG",
                dag_id=dag_id,
                status=response.status_code,
                body=response.text[:200],
            )
            return False
    except requests.RequestException as exc:
        log.error("Network error triggering DAG", dag_id=dag_id, error=str(exc))
        return False


def maybe_trigger_alert(drift_score: float, dataset_drift: bool) -> None:
    """
    Fire retraining trigger if drift exceeds the configured threshold.
    This is called at the end of every drift_check_dag run.
    """
    if dataset_drift or drift_score >= settings.drift_threshold:
        log.warning(
            "Drift threshold exceeded — triggering retraining",
            drift_score=drift_score,
            threshold=settings.drift_threshold,
            dataset_drift=dataset_drift,
        )
        trigger_retrain_dag()
    else:
        log.info(
            "Drift within acceptable range — no action",
            drift_score=drift_score,
            threshold=settings.drift_threshold,
        )
