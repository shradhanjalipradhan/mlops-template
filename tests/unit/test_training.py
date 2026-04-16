"""Unit tests for training and evaluation utilities."""
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from src.training.evaluate import compute_metrics
from src.features.pipeline import build_preprocessor, split_xy, add_engineered_features


class TestComputeMetrics:
    @pytest.fixture
    def fitted_model(self, train_df):
        X, y = split_xy(add_engineered_features(train_df))
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", DummyClassifier(strategy="stratified", random_state=0)),
        ])
        pipe.fit(X, y)
        return pipe

    def test_returns_all_expected_keys(self, fitted_model, val_df):
        X_val, y_val = split_xy(add_engineered_features(val_df))
        metrics = compute_metrics(fitted_model, X_val, y_val)
        expected = {
            "val_roc_auc", "val_avg_precision",
            "val_f1", "val_precision", "val_recall",
            "val_support_pos", "val_support_neg",
        }
        assert expected.issubset(set(metrics.keys()))

    def test_metrics_in_valid_range(self, fitted_model, val_df):
        X_val, y_val = split_xy(add_engineered_features(val_df))
        metrics = compute_metrics(fitted_model, X_val, y_val)
        for key in ["val_roc_auc", "val_avg_precision", "val_f1", "val_precision", "val_recall"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_supports_sum_to_total(self, fitted_model, val_df):
        X_val, y_val = split_xy(add_engineered_features(val_df))
        metrics = compute_metrics(fitted_model, X_val, y_val)
        assert metrics["val_support_pos"] + metrics["val_support_neg"] == len(val_df)
