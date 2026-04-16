#!/usr/bin/env python3
"""Generate synthetic churn data and save to data/raw/churn.csv."""
import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import generate_churn_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic churn data")
    parser.add_argument("--n", type=int, default=5000, help="Number of rows (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    args = parser.parse_args()

    df = generate_churn_dataset(n_rows=args.n, seed=args.seed)
    path = save_dataset(df, output_dir=args.output_dir)
    print(f"✓  Dataset written: {path}  ({len(df)} rows, {df['churned'].mean():.1%} churn rate)")


if __name__ == "__main__":
    main()
