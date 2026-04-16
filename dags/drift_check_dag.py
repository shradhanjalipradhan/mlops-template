"""Airflow DAG: hourly drift check using Evidently AI."""
from datetime import datetime, timedelta
from airflow.decorators import dag, task


@dag(
    dag_id="drift_check_dag",
    schedule="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["monitoring", "drift"],
    doc_md="""
    ## Drift Check DAG

    Runs hourly. Reads the last N predictions from the SQLite prediction log,
    compares feature distributions to the reference dataset using Evidently AI,
    writes a drift score to the drift_log table, saves an HTML report, and
    triggers the retrain_dag if drift exceeds the configured threshold.
    """,
)
def drift_check():

    @task
    def load_recent_predictions() -> str:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import sqlite3
        import pandas as pd
        from src.utils.config import settings

        conn = sqlite3.connect(settings.db_path)
        df = pd.read_sql(
            f"""
            SELECT age, tenure_months, monthly_charge, num_products,
                   contract_type, payment_method, label, churn_probability
            FROM predictions
            ORDER BY created_at DESC
            LIMIT {settings.drift_window_rows}
            """,
            conn,
        )
        conn.close()

        if len(df) < 50:
            raise ValueError(
                f"Not enough predictions for drift analysis: {len(df)} rows (need ≥ 50)"
            )

        out = "/tmp/current_predictions.parquet"
        df.to_parquet(out, index=False)
        return out

    @task
    def compute_drift(current_path: str) -> dict:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.monitoring.drift_detector import run_drift_report
        from src.utils.config import settings

        current_df = pd.read_parquet(current_path)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")

        return run_drift_report(
            reference_path=settings.reference_data_path,
            current_df=current_df,
            output_dir="/opt/airflow/monitoring/reports",
            report_name=f"drift_{timestamp}",
        )

    @task
    def persist_and_alert(drift_result: dict) -> bool:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import sqlite3
        from src.utils.config import settings
        from src.monitoring.alerting import maybe_trigger_alert
        from src.serving.metrics import DRIFT_SCORE, DATASET_DRIFT_FLAG

        conn = sqlite3.connect(settings.db_path)
        conn.execute(
            """INSERT INTO drift_log (created_at, drift_score, dataset_drift, report_path)
               VALUES (datetime('now'), ?, ?, ?)""",
            (
                drift_result["drift_score"],
                int(drift_result["dataset_drift"]),
                drift_result["report_path"],
            ),
        )
        conn.commit()
        conn.close()

        DRIFT_SCORE.set(drift_result["drift_score"])
        DATASET_DRIFT_FLAG.set(1 if drift_result["dataset_drift"] else 0)

        maybe_trigger_alert(
            drift_score=drift_result["drift_score"],
            dataset_drift=drift_result["dataset_drift"],
        )

        return drift_result["dataset_drift"]

    current = load_recent_predictions()
    drift = compute_drift(current)
    persist_and_alert(drift)


drift_check()
