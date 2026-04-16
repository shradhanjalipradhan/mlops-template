from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"

    # Model
    model_name: str = "churn_model"
    model_stage: str = "Production"

    # Database
    db_path: str = "/data/predictions.db"
    reference_data_path: str = "/data/reference/reference.parquet"

    # Logging
    log_level: str = "INFO"

    # Airflow
    airflow_base_url: str = "http://airflow-webserver:8080"
    airflow_user: str = "admin"
    airflow_password: str = "admin"

    # Drift
    drift_threshold: float = 0.3
    drift_window_rows: int = 500


settings = Settings()
