#!/usr/bin/env python3
"""Send N prediction requests to the inference API to generate realistic traffic."""
import sys
import argparse
import time
import numpy as np
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


CONTRACT_TYPES = ["month-to-month", "one_year", "two_year"]
PAYMENT_METHODS = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]


def generate_payload(rng: np.random.Generator) -> dict:
    return {
        "age": float(rng.integers(25, 76)),
        "tenure_months": float(rng.integers(1, 73)),
        "monthly_charge": float(rng.uniform(20, 120)),
        "num_products": float(rng.integers(1, 6)),
        "contract_type": str(rng.choice(CONTRACT_TYPES)),
        "payment_method": str(rng.choice(PAYMENT_METHODS)),
    }


def run_load_test(n: int, base_url: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    url = f"{base_url}/predict"

    success, errors = 0, 0
    latencies = []
    start = time.perf_counter()

    print(f"Sending {n} requests to {url}...")

    for i in range(n):
        payload = generate_payload(rng)
        try:
            t0 = time.perf_counter()
            r = requests.post(url, json=payload, timeout=5)
            latency = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                success += 1
                latencies.append(latency)
            else:
                errors += 1
                if errors <= 5:
                    print(f"  Error {r.status_code}: {r.text[:100]}")
        except requests.RequestException as e:
            errors += 1
            if errors <= 5:
                print(f"  Request failed: {e}")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} sent  ({success} ok, {errors} errors)")

    total = time.perf_counter() - start
    latencies_arr = np.array(latencies) if latencies else np.array([0])

    print(f"\n{'='*40}")
    print(f"  Total requests : {n}")
    print(f"  Successful     : {success}")
    print(f"  Errors         : {errors}")
    print(f"  Total time     : {total:.2f}s")
    print(f"  Throughput     : {success / total:.1f} req/s")
    print(f"  Latency p50    : {np.percentile(latencies_arr, 50):.1f} ms")
    print(f"  Latency p95    : {np.percentile(latencies_arr, 95):.1f} ms")
    print(f"  Latency p99    : {np.percentile(latencies_arr, 99):.1f} ms")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description="Load test the inference API")
    parser.add_argument("--n", type=int, default=500, help="Number of requests")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    run_load_test(n=args.n, base_url=args.url, seed=args.seed)


if __name__ == "__main__":
    main()
