"""Integration tests for the FastAPI inference endpoints.

These tests run against a live service. In CI, docker-compose.ci.yml
boots MLflow + the serving container before pytest runs.
"""
import pytest
import os
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session", autouse=True)
def wait_for_api():
    """Wait until the API is responsive before running tests."""
    import time
    for _ in range(30):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=3)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    pytest.skip("API not reachable — skipping integration tests")


class TestHealthEndpoint:
    def test_returns_200(self):
        r = requests.get(f"{API_BASE}/health")
        assert r.status_code == 200

    def test_response_has_status_ok(self):
        r = requests.get(f"{API_BASE}/health")
        body = r.json()
        assert body["status"] == "ok"


class TestPredictEndpoint:
    VALID_PAYLOAD = {
        "age": 45.0,
        "tenure_months": 12.0,
        "monthly_charge": 89.5,
        "num_products": 2.0,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
    }

    def test_returns_200_on_valid_input(self):
        r = requests.post(f"{API_BASE}/predict", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self):
        r = requests.post(f"{API_BASE}/predict", json=self.VALID_PAYLOAD)
        body = r.json()
        assert "label" in body
        assert "churn_probability" in body
        assert "latency_ms" in body
        assert body["label"] in ("churn", "no_churn")
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["latency_ms"] >= 0

    def test_returns_422_on_missing_field(self):
        bad = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "age"}
        r = requests.post(f"{API_BASE}/predict", json=bad)
        assert r.status_code == 422

    def test_returns_422_on_age_out_of_range(self):
        bad = {**self.VALID_PAYLOAD, "age": 5.0}
        r = requests.post(f"{API_BASE}/predict", json=bad)
        assert r.status_code == 422

    def test_multiple_requests_are_consistent(self):
        """Same input should return the same label (deterministic model)."""
        labels = set()
        for _ in range(5):
            r = requests.post(f"{API_BASE}/predict", json=self.VALID_PAYLOAD)
            labels.add(r.json()["label"])
        assert len(labels) == 1, "Model is non-deterministic for identical inputs"


class TestMetricsEndpoint:
    def test_metrics_endpoint_reachable(self):
        r = requests.get(f"{API_BASE}/metrics")
        assert r.status_code == 200

    def test_metrics_contain_inference_histogram(self):
        # Ensure at least one prediction has been made before checking
        requests.post(f"{API_BASE}/predict", json=TestPredictEndpoint.VALID_PAYLOAD)
        r = requests.get(f"{API_BASE}/metrics")
        assert "inference_latency_seconds" in r.text

    def test_metrics_contain_prediction_counter(self):
        requests.post(f"{API_BASE}/predict", json=TestPredictEndpoint.VALID_PAYLOAD)
        r = requests.get(f"{API_BASE}/metrics")
        assert "prediction_total" in r.text
