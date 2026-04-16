"""Abstraction over MinIO / local filesystem artifact storage."""
import os
from pathlib import Path
from src.utils.logging import get_logger

log = get_logger(__name__)


def get_artifact_uri(relative_path: str, use_s3: bool = False) -> str:
    """Return an artifact URI suitable for MLflow artifact logging."""
    if use_s3:
        bucket = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")
        return f"s3://{bucket}/{relative_path}"
    local_root = Path(os.getenv("LOCAL_ARTIFACT_ROOT", "/tmp/mlflow-artifacts"))
    local_root.mkdir(parents=True, exist_ok=True)
    return str(local_root / relative_path)


def ensure_bucket(bucket_name: str) -> None:
    """Create MinIO bucket if it doesn't exist. Silently skips in local mode."""
    try:
        import boto3
        from botocore.exceptions import ClientError
        from src.utils.config import settings

        s3 = boto3.client(
            "s3",
            endpoint_url=settings.mlflow_s3_endpoint_url,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        try:
            s3.head_bucket(Bucket=bucket_name)
            log.info("Bucket already exists", bucket=bucket_name)
        except ClientError:
            s3.create_bucket(Bucket=bucket_name)
            log.info("Bucket created", bucket=bucket_name)
    except Exception as exc:
        log.warning("Could not ensure bucket — running in local mode?", error=str(exc))
