"""Async prediction logger — writes every inference to SQLite."""
import sqlite3
import threading
from datetime import datetime
from src.serving.schemas import PredictRequest
from src.utils.logging import get_logger

log = get_logger(__name__)

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    age             REAL,
    tenure_months   REAL,
    monthly_charge  REAL,
    num_products    REAL,
    contract_type   TEXT,
    payment_method  TEXT,
    label           TEXT,
    churn_probability REAL
);
"""

CREATE_DRIFT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS drift_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    drift_score     REAL,
    dataset_drift   INTEGER,
    report_path     TEXT
);
"""


class PredictionLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(CREATE_PREDICTIONS_TABLE)
            conn.execute(CREATE_DRIFT_LOG_TABLE)
        log.info("Prediction database initialised", db_path=self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def log(self, request: PredictRequest, label: str, probability: float) -> None:
        row = (
            datetime.utcnow().isoformat(),
            request.age,
            request.tenure_months,
            request.monthly_charge,
            request.num_products,
            request.contract_type,
            request.payment_method,
            label,
            probability,
        )
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO predictions
                    (created_at, age, tenure_months, monthly_charge,
                     num_products, contract_type, payment_method, label, churn_probability)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    row,
                )

    def close(self) -> None:
        log.info("PredictionLogger closed")
