"""Airflow DAG: train XGBoost model and register to MLflow model registry."""
from datetime import datetime, timedelta
from airflow.decorators import dag, task


@dag(
    dag_id="train_register_dag",
    schedule=None,  # triggered manually or by retrain_dag
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["training", "mlflow"],
    doc_md="""
    ## Train & Register DAG

    Triggered manually or by the retrain_dag. Trains an XGBoost churn model,
    logs all metrics and artifacts to MLflow, and promotes to Production stage
    if val_roc_auc >= min_auc (configured in configs/training_config.yaml).
    """,
)
def train_register():

    @task
    def run_training() -> str:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.training.train import train
        import yaml
        with open("configs/training_config.yaml") as f:
            cfg = yaml.safe_load(f)
        run_id = train(
            train_path="data/processed/train.parquet",
            val_path="data/processed/val.parquet",
            experiment_name=cfg.get("experiment_name", "churn_prediction"),
            min_auc=cfg.get("min_auc", 0.75),
        )
        return run_id

    @task
    def notify(run_id: str) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.utils.logging import get_logger
        log = get_logger("train_register_dag")
        log.info("Training DAG complete", run_id=run_id)

    run_id = run_training()
    notify(run_id)


train_register()
