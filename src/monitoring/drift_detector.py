"""Evidently-based drift detection for feature and prediction distributions."""
from pathlib import Path
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from src.utils.logging import get_logger

log = get_logger(__name__)

NUMERIC_FEATURES = ["age", "tenure_months", "monthly_charge", "num_products"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
TARGET = "churned"
PREDICTION_COL = "label"


def run_drift_report(
    reference_path: str,
    current_df: pd.DataFrame,
    output_dir: str,
    report_name: str,
) -> dict:
    """
    Compare current_df against the reference dataset.

    Writes an HTML report to output_dir and returns a summary dict with:
        drift_score     — share of drifted columns (0.0 – 1.0)
        dataset_drift   — True if overall drift is detected
        report_path     — path to the HTML report
    """
    reference_df = pd.read_parquet(reference_path)

    # Align prediction column name — inference logs use "label" as a string,
    # reference data uses "churned" as an int. Map both to numeric for comparison.
    current_df = current_df.copy()
    if "label" in current_df.columns:
        current_df["churned_pred"] = (current_df["label"] == "churn").astype(int)
    else:
        current_df["churned_pred"] = current_df.get("churned", 0)

    column_mapping = ColumnMapping(
        target=None,                    # no ground truth at inference time
        prediction="churned_pred",
        numerical_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = str(Path(output_dir) / f"{report_name}.html")
    report.save_html(report_path)

    result = report.as_dict()
    drift_summary = result["metrics"][0]["result"]
    drift_score = float(drift_summary.get("share_of_drifted_columns", 0.0))
    dataset_drift = bool(drift_summary.get("dataset_drift", False))

    log.info(
        "Drift report complete",
        drift_score=f"{drift_score:.3f}",
        dataset_drift=dataset_drift,
        report=report_path,
        n_current=len(current_df),
        n_reference=len(reference_df),
    )

    return {
        "drift_score": drift_score,
        "dataset_drift": dataset_drift,
        "report_path": report_path,
    }
