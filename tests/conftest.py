"""Shared pytest fixtures for unit and integration tests."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


# ── Synthetic data fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Minimal valid raw churn DataFrame for testing."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "age": rng.integers(25, 76, n).astype(float),
        "tenure_months": rng.integers(1, 73, n).astype(float),
        "monthly_charge": rng.uniform(20, 120, n),
        "num_products": rng.integers(1, 6, n).astype(float),
        "contract_type": rng.choice(["month-to-month", "one_year", "two_year"], n),
        "payment_method": rng.choice(
            ["electronic_check", "mailed_check", "bank_transfer", "credit_card"], n
        ),
        "churned": rng.integers(0, 2, n),
    })


@pytest.fixture(scope="session")
def train_df(raw_df) -> pd.DataFrame:
    return raw_df.iloc[:140].reset_index(drop=True)


@pytest.fixture(scope="session")
def val_df(raw_df) -> pd.DataFrame:
    return raw_df.iloc[140:170].reset_index(drop=True)


@pytest.fixture(scope="session")
def test_df(raw_df) -> pd.DataFrame:
    return raw_df.iloc[170:].reset_index(drop=True)


@pytest.fixture
def single_request_dict() -> dict:
    return {
        "age": 45.0,
        "tenure_months": 12.0,
        "monthly_charge": 89.5,
        "num_products": 2.0,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
    }


# ── Mock MLflow ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_mlflow(monkeypatch):
    mock = MagicMock()
    mock.start_run.return_value.__enter__ = MagicMock(
        return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
    )
    mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("src.training.train.mlflow", mock)
    return mock


# ── Temp SQLite DB ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path) -> str:
    return str(tmp_path / "test_predictions.db")
