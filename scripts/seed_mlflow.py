#!/usr/bin/env python3
"""
Seed script: generate data, run full training pipeline, register first model.
Run this once after `make up` to get a Production model deployed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import generate_churn_dataset, save_dataset
from src.data.ingestion import clean, split, save_splits, save_reference
from src.data.validation import validate
from src.training.train import train
from src.utils.logging import get_logger

log = get_logger("seed_mlflow")


def main():
    log.info("=== Seeding MLflow with first trained model ===")

    # Step 1 — generate data
    log.info("Generating synthetic data...")
    df = generate_churn_dataset(n_rows=5000, seed=42)
    save_dataset(df, output_dir="data/raw")

    # Step 2 — clean + validate
    log.info("Cleaning and validating...")
    cleaned = clean(df)
    validate(cleaned)

    # Step 3 — split + save reference
    log.info("Splitting data...")
    train_df, val_df, test_df = split(cleaned)
    save_splits(train_df, val_df, test_df, output_dir="data/processed")
    save_reference(train_df, output_dir="data/reference")

    # Step 4 — train and register
    log.info("Training model...")
    run_id = train(
        train_path="data/processed/train.parquet",
        val_path="data/processed/val.parquet",
        experiment_name="churn_prediction",
        min_auc=0.0,  # always register the first seed model
    )

    log.info("=== Seed complete ===", run_id=run_id)
    print(f"\n✓  Model trained and registered. Run ID: {run_id}")
    print("  MLflow UI → http://localhost:5000")
    print("  API       → http://localhost:8000/docs")


if __name__ == "__main__":
    main()
