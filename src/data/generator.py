"""Generate reproducible synthetic churn data for development and testing."""
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logging import get_logger

log = get_logger(__name__)

CONTRACT_TYPES = ["month-to-month", "one_year", "two_year"]
PAYMENT_METHODS = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]


def generate_churn_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic churn dataset with realistic feature correlations.

    Features:
        age                 — customer age (25–75)
        tenure_months       — months with company (1–72)
        monthly_charge      — monthly bill in USD (20–120)
        num_products        — number of products subscribed (1–5)
        contract_type       — contract length category
        payment_method      — payment channel

    Target:
        churned (bool)      — 1 if customer left, 0 if retained
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(25, 76, n_rows).astype(float)
    tenure = rng.integers(1, 73, n_rows).astype(float)
    monthly_charge = rng.uniform(20, 120, n_rows)
    num_products = rng.integers(1, 6, n_rows).astype(float)
    contract_type = rng.choice(CONTRACT_TYPES, n_rows, p=[0.5, 0.3, 0.2])
    payment_method = rng.choice(PAYMENT_METHODS, n_rows)

    # Churn probability driven by business logic
    logit = (
        -3.0
        + 0.03 * monthly_charge       # higher bill → more likely to churn
        - 0.04 * tenure               # longer tenure → less likely to churn
        - 0.5 * num_products           # more products → stickier
        + 0.8 * (contract_type == "month-to-month").astype(float)
        + 0.3 * (payment_method == "electronic_check").astype(float)
        + rng.normal(0, 0.5, n_rows)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churned = (rng.uniform(0, 1, n_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure,
        "monthly_charge": monthly_charge,
        "num_products": num_products,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "churned": churned,
    })

    log.info(
        "Synthetic dataset generated",
        n_rows=n_rows,
        churn_rate=f"{churned.mean():.2%}",
        seed=seed,
    )
    return df


def save_dataset(df: pd.DataFrame, output_dir: str = "data/raw") -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "churn.csv"
    df.to_csv(path, index=False)
    log.info("Dataset saved", path=str(path), rows=len(df))
    return path


if __name__ == "__main__":
    df = generate_churn_dataset()
    save_dataset(df)
