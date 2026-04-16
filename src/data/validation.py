"""Data validation using Great Expectations."""
import pandas as pd
from src.utils.logging import get_logger

log = get_logger(__name__)


def validate(df: pd.DataFrame) -> bool:
    """
    Run schema + quality checks on a DataFrame.
    Returns True if all checks pass, raises ValueError on failure.

    Uses manual checks rather than the full GE suite to keep the dependency
    footprint lean. Swap this function body for a GE Checkpoint if you need
    richer HTML reports.
    """
    errors: list[str] = []

    # Schema checks
    required = {
        "age": "float64",
        "tenure_months": "float64",
        "monthly_charge": "float64",
        "num_products": "float64",
        "contract_type": "object",
        "payment_method": "object",
        "churned": "int64",
    }
    for col, expected_dtype in required.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
        elif not str(df[col].dtype).startswith(expected_dtype.rstrip("64")):
            errors.append(f"Column {col}: expected {expected_dtype}, got {df[col].dtype}")

    # Null rate checks (max 5% nulls per column)
    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate > 0.05:
            errors.append(f"Column {col}: null rate {null_rate:.1%} exceeds 5% threshold")

    # Value range checks
    if "age" in df.columns and not df["age"].between(18, 100).all():
        errors.append("Column age: values outside [18, 100]")
    if "monthly_charge" in df.columns and not df["monthly_charge"].between(0, 500).all():
        errors.append("Column monthly_charge: values outside [0, 500]")
    if "churned" in df.columns and not df["churned"].isin([0, 1]).all():
        errors.append("Column churned: expected binary values {0, 1}")

    # Minimum row count
    if len(df) < 100:
        errors.append(f"Too few rows: {len(df)} (minimum 100)")

    if errors:
        msg = f"Validation failed with {len(errors)} error(s):\n" + "\n".join(f"  • {e}" for e in errors)
        log.error("Data validation failed", errors=errors)
        raise ValueError(msg)

    log.info("Data validation passed", rows=len(df), columns=list(df.columns))
    return True
