"""Unit tests for the serving layer — schemas, logger, metrics."""
import pytest
import sqlite3
from unittest.mock import MagicMock, patch

from src.serving.schemas import PredictRequest, PredictResponse
from src.serving.logger import PredictionLogger


class TestSchemas:
    def test_valid_request_parses(self, single_request_dict):
        req = PredictRequest(**single_request_dict)
        assert req.age == 45.0
        assert req.contract_type == "month-to-month"

    def test_age_out_of_range_raises(self, single_request_dict):
        from pydantic import ValidationError
        bad = {**single_request_dict, "age": 10.0}
        with pytest.raises(ValidationError):
            PredictRequest(**bad)

    def test_response_schema(self):
        resp = PredictResponse(label="churn", churn_probability=0.82, latency_ms=3.4)
        assert resp.label == "churn"
        assert 0 <= resp.churn_probability <= 1


class TestPredictionLogger:
    def test_creates_tables_on_init(self, tmp_db):
        logger = PredictionLogger(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "predictions" in tables
        assert "drift_log" in tables
        conn.close()
        logger.close()

    def test_log_inserts_row(self, tmp_db, single_request_dict):
        from src.serving.schemas import PredictRequest
        logger = PredictionLogger(db_path=tmp_db)
        req = PredictRequest(**single_request_dict)
        logger.log(request=req, label="churn", probability=0.75)

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute("SELECT * FROM predictions").fetchall()
        conn.close()
        logger.close()

        assert len(rows) == 1
        # label is stored correctly
        assert rows[0][8] == "churn"

    def test_log_is_thread_safe(self, tmp_db, single_request_dict):
        import threading
        from src.serving.schemas import PredictRequest

        logger = PredictionLogger(db_path=tmp_db)
        req = PredictRequest(**single_request_dict)

        def write():
            for _ in range(20):
                logger.log(request=req, label="no_churn", probability=0.2)

        threads = [threading.Thread(target=write) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        logger.close()

        assert count == 100  # 5 threads × 20 writes
