"""Airflow DAG: ingest raw data, validate, split, and save reference dataset."""
from datetime import datetime, timedelta
from airflow.decorators import dag, task


@dag(
    dag_id="ingest_validate_dag",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["data", "ingestion"],
    doc_md="""
    ## Ingest & Validate DAG

    Runs daily. Loads raw CSV from `data/raw/churn.csv`, validates the schema
    and quality constraints, splits into train/val/test, and saves a reference
    dataset for Evidently drift detection.
    """,
)
def ingest_validate():

    @task
    def load_and_clean() -> str:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.data.ingestion import load_raw, clean
        df = load_raw("data/raw/churn.csv")
        df = clean(df)
        out = "/tmp/cleaned.parquet"
        df.to_parquet(out, index=False)
        return out

    @task
    def validate(cleaned_path: str) -> str:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.data.validation import validate as run_validate
        df = pd.read_parquet(cleaned_path)
        run_validate(df)
        return cleaned_path

    @task
    def split_and_save(cleaned_path: str) -> dict:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.data.ingestion import split, save_splits, save_reference
        df = pd.read_parquet(cleaned_path)
        train, val, test = split(df)
        save_splits(train, val, test, output_dir="data/processed")
        save_reference(train, output_dir="data/reference")
        return {"train": len(train), "val": len(val), "test": len(test)}

    cleaned = load_and_clean()
    validated = validate(cleaned)
    split_and_save(validated)


ingest_validate()
