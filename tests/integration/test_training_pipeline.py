"""Integration test for the end-to-end training pipeline.

Runs a full train → evaluate → register cycle against a live MLflow server.
Skipped automatically if MLFLOW_TRACKING_URI is not reachable.
"""
import pytest
import os
import requests

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@pytest.fixture(scope="module", autouse=True)
def require_mlflow():
    try:
        r = requests.get(f"{MLFLOW_URI}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("MLflow not healthy — skipping training integration tests")
    except requests.ConnectionError:
        pytest.skip("MLflow not reachable — skipping training integration tests")


def test_full_training_pipeline_produces_run(tmp_path, raw_df):
    """Train a model end-to-end and verify an MLflow run is created."""
    import mlflow
    from src.data.ingestion import clean, split, save_splits, save_reference
    from src.training.train import train

    # Prepare data
    cleaned = clean(raw_df)
    tr, val, te = split(cleaned)

    train_path = str(tmp_path / "train.parquet")
    val_path = str(tmp_path / "val.parquet")
    tr.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    mlflow.set_tracking_uri(MLFLOW_URI)
    run_id = train(
        train_path=train_path,
        val_path=val_path,
        experiment_name="integration_test",
        min_auc=0.0,  # always register in tests
    )

    assert run_id is not None
    assert len(run_id) > 0

    # Verify the run exists in MLflow
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert "val_roc_auc" in run.data.metrics
    assert run.data.metrics["val_roc_auc"] > 0.0
