"""Model evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from src.utils.logging import get_logger

log = get_logger(__name__)


def compute_metrics(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return a flat dict of metrics suitable for mlflow.log_metrics."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "val_roc_auc": float(roc_auc_score(y, proba)),
        "val_avg_precision": float(average_precision_score(y, proba)),
        "val_f1": float(f1_score(y, preds, zero_division=0)),
        "val_precision": float(precision_score(y, preds, zero_division=0)),
        "val_recall": float(recall_score(y, preds, zero_division=0)),
        "val_support_pos": int(y.sum()),
        "val_support_neg": int((y == 0).sum()),
    }

    log.info(
        "Evaluation complete",
        roc_auc=f"{metrics['val_roc_auc']:.4f}",
        f1=f"{metrics['val_f1']:.4f}",
        precision=f"{metrics['val_precision']:.4f}",
        recall=f"{metrics['val_recall']:.4f}",
    )
    return metrics
