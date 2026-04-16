"""Main training entry point."""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from pathlib import Path

from src.features.pipeline import build_preprocessor, split_xy, add_engineered_features
from src.training.evaluate import compute_metrics
from src.training.register import register_model
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


def train(
    train_path: str = "data/processed/train.parquet",
    val_path: str = "data/processed/val.parquet",
    experiment_name: str = "churn_prediction",
    min_auc: float = 0.75,
) -> str:
    """
    Train an XGBoost churn model, log to MLflow, register if AUC > min_auc.
    Returns the MLflow run_id.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": float((y_train == 0).sum() / (y_train == 1).sum()),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(**params)),
    ])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info("Training started", run_id=run_id, params=params)
        mlflow.log_params(params)

        # Cross-validation on training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mlflow.log_metric("cv_auc_mean", float(np.mean(cv_aucs)))
        mlflow.log_metric("cv_auc_std", float(np.std(cv_aucs)))
        log.info("CV complete", cv_auc_mean=float(np.mean(cv_aucs)), cv_auc_std=float(np.std(cv_aucs)))

        # Final fit on full training set
        model.fit(X_train, y_train)

        # Validation metrics
        metrics = compute_metrics(model, X_val, y_val)
        mlflow.log_metrics(metrics)
        log.info("Validation metrics", **metrics)

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=settings.model_name,
            input_example=X_val.iloc[:3],
        )

        # Conditional registration to Production
        if metrics["val_roc_auc"] >= min_auc:
            register_model(run_id=run_id, auc=metrics["val_roc_auc"])
        else:
            log.warning(
                "Model did not meet quality threshold — not promoted",
                val_roc_auc=metrics["val_roc_auc"],
                min_auc=min_auc,
            )

    return run_id


if __name__ == "__main__":
    run_id = train()
    log.info("Training complete", run_id=run_id)
