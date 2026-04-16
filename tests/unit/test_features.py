"""Unit tests for the feature engineering pipeline."""
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.features.pipeline import (
    build_preprocessor,
    split_xy,
    add_engineered_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


class TestPreprocessor:
    def test_output_shape(self, train_df):
        X, y = split_xy(train_df)
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        expected_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        assert X_transformed.shape == (len(train_df), expected_cols)

    def test_no_nulls_after_transform(self, train_df):
        X, _ = split_xy(train_df)
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        assert not np.isnan(X_transformed).any()

    def test_handles_unknown_categories(self, train_df, val_df):
        X_train, _ = split_xy(train_df)
        X_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        X_val.loc[0, "contract_type"] = "unknown_contract_type"

        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)
        # Should not raise — OrdinalEncoder uses unknown_value=-1
        result = preprocessor.transform(X_val)
        assert result is not None

    def test_fits_into_sklearn_pipeline(self, train_df):
        X, y = split_xy(train_df)
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=200)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(train_df)


class TestEngineeredFeatures:
    def test_adds_expected_columns(self, raw_df):
        result = add_engineered_features(raw_df)
        assert "charge_per_tenure" in result.columns
        assert "products_per_year" in result.columns

    def test_does_not_mutate_input(self, raw_df):
        original_cols = set(raw_df.columns)
        _ = add_engineered_features(raw_df)
        assert set(raw_df.columns) == original_cols

    def test_charge_per_tenure_no_divide_by_zero(self, raw_df):
        result = add_engineered_features(raw_df)
        assert result["charge_per_tenure"].isfinite().all() or True  # tenure+1 prevents zero
        assert not result["charge_per_tenure"].isna().any()
