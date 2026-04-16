"""Load, clean, and split raw churn data."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logging import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = {
    "age", "tenure_months", "monthly_charge",
    "num_products", "contract_type", "payment_method", "churned",
}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Raw data loaded", path=path, shape=df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=list(REQUIRED_COLUMNS))

    # Clamp numeric ranges to valid domain
    df = df[df["age"].between(18, 100)]
    df = df[df["tenure_months"].between(0, 240)]
    df = df[df["monthly_charge"].between(0, 500)]
    df = df[df["num_products"].between(1, 20)]

    log.info("Data cleaned", rows_before=before, rows_after=len(df), dropped=before - len(df))
    return df.reset_index(drop=True)


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["churned"])
    val_fraction = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_fraction, random_state=seed, stratify=train_val["churned"])

    log.info("Data split", train=len(train), val=len(val), test=len(test))
    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str = "data/processed",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.to_parquet(f"{output_dir}/train.parquet", index=False)
    val.to_parquet(f"{output_dir}/val.parquet", index=False)
    test.to_parquet(f"{output_dir}/test.parquet", index=False)
    log.info("Splits saved", output_dir=output_dir)


def save_reference(train: pd.DataFrame, output_dir: str = "data/reference") -> None:
    """Save training data as the Evidently reference dataset."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.to_parquet(f"{output_dir}/reference.parquet", index=False)
    log.info("Reference dataset saved", rows=len(train))
