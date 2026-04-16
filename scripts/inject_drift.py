#!/usr/bin/env python3
"""
Inject synthetic drift into the prediction log to trigger the drift detector.

Simulates a real-world scenario where the input feature distribution shifts:
  - monthly_charge is inflated by +40 (price hike scenario)
  - tenure_months is compressed (new customer surge scenario)

Run this after `make load-test` to demonstrate the full drift → retrain loop.
"""
import sys
import sqlite3
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("inject_drift")


def inject(db_path: str, n_rows: int = 300, seed: int = 99) -> None:
    rng = np.random.default_rng(seed)

    log.info("Injecting drifted predictions into prediction log", n_rows=n_rows, db_path=db_path)

    conn = sqlite3.connect(db_path)

    # Drifted distribution: high monthly_charge, low tenure
    rows = []
    for i in range(n_rows):
        # Bimodal shift: charge 40-60 units higher than training distribution
        monthly_charge = float(rng.uniform(70, 160))
        tenure = float(rng.integers(1, 10))          # very short tenure (new users)
        age = float(rng.integers(18, 40))            # younger cohort

        rows.append((
            (datetime.utcnow() - timedelta(minutes=n_rows - i)).isoformat(),
            age,
            tenure,
            monthly_charge,
            float(rng.integers(1, 3)),
            rng.choice(["month-to-month", "month-to-month", "one_year"]),  # skewed
            rng.choice(["electronic_check", "electronic_check", "mailed_check"]),  # skewed
            "churn",
            float(rng.uniform(0.6, 0.99)),
        ))

    conn.executemany(
        """INSERT INTO predictions
           (created_at, age, tenure_months, monthly_charge, num_products,
            contract_type, payment_method, label, churn_probability)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    conn.close()

    log.info("Drift injection complete", rows_inserted=n_rows)
    print(f"\n✓  Injected {n_rows} drifted predictions into {db_path}")
    print("  Next drift_check_dag run will detect drift and trigger retraining.")
    print("  To trigger manually: make trigger-drift")


def main():
    parser = argparse.ArgumentParser(description="Inject synthetic drift into prediction log")
    parser.add_argument("--n", type=int, default=300, help="Number of drifted rows to inject")
    parser.add_argument("--db", default=settings.db_path, help="Path to predictions SQLite DB")
    args = parser.parse_args()
    inject(db_path=args.db, n_rows=args.n)


if __name__ == "__main__":
    main()
