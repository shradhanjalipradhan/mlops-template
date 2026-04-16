"""Feature engineering pipeline — sklearn ColumnTransformer."""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.utils.logging import get_logger

log = get_logger(__name__)

NUMERIC_FEATURES = ["age", "tenure_months", "monthly_charge", "num_products"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
TARGET = "churned"


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an unfitted sklearn ColumnTransformer.

    Numeric: median impute → standard scale
    Categorical: constant impute → ordinal encode
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features before the preprocessor runs."""
    df = df.copy()
    # Charge per month of tenure — high = potentially price-sensitive recent customer
    df["charge_per_tenure"] = df["monthly_charge"] / (df["tenure_months"] + 1)
    # Products per year of tenure
    df["products_per_year"] = df["num_products"] / (df["tenure_months"] / 12 + 1)
    return df
