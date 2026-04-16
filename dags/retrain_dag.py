"""Airflow DAG: triggered retraining pipeline (ingest → train → register)."""
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


@dag(
    dag_id="retrain_dag",
    schedule=None,  # only triggered — never scheduled
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["retraining", "drift"],
    doc_md="""
    ## Retrain DAG

    Triggered by the drift_check_dag when feature drift exceeds the threshold.
    Chains the full ingest → validate → train → register pipeline.
    A new model is promoted to Production only if it beats the quality threshold.
    """,
)
def retrain():

    @task
    def log_trigger(**context) -> dict:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.utils.logging import get_logger
        log = get_logger("retrain_dag")
        conf = context.get("dag_run").conf or {}
        log.info("Retraining triggered", conf=conf)
        return conf

    trigger_ingest = TriggerDagRunOperator(
        task_id="trigger_ingest_validate",
        trigger_dag_id="ingest_validate_dag",
        wait_for_completion=True,
        poke_interval=30,
        allowed_states=["success"],
        failed_states=["failed"],
    )

    trigger_train = TriggerDagRunOperator(
        task_id="trigger_train_register",
        trigger_dag_id="train_register_dag",
        wait_for_completion=True,
        poke_interval=30,
        allowed_states=["success"],
        failed_states=["failed"],
    )

    @task
    def log_complete() -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.utils.logging import get_logger
        log = get_logger("retrain_dag")
        log.info("Retraining pipeline complete")

    conf = log_trigger()
    conf >> trigger_ingest >> trigger_train >> log_complete()


retrain()
