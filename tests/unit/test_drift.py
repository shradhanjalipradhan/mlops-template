"""Unit tests for drift detection."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestDriftDetector:
    @pytest.fixture
    def reference_parquet(self, tmp_path, raw_df):
        path = tmp_path / "reference.parquet"
        raw_df.to_parquet(path, index=False)
        return str(path)

    @pytest.fixture
    def current_df(self, raw_df):
        """Current data with no significant drift."""
        return raw_df.copy()

    @pytest.fixture
    def drifted_df(self, raw_df):
        """Current data with obvious distribution shift."""
        df = raw_df.copy()
        df["monthly_charge"] = df["monthly_charge"] + 60.0  # large shift
        return df

    def test_run_drift_report_returns_expected_keys(
        self, reference_parquet, current_df, tmp_path
    ):
        from src.monitoring.drift_detector import run_drift_report

        result = run_drift_report(
            reference_path=reference_parquet,
            current_df=current_df,
            output_dir=str(tmp_path),
            report_name="test_report",
        )
        assert "drift_score" in result
        assert "dataset_drift" in result
        assert "report_path" in result
        assert 0.0 <= result["drift_score"] <= 1.0
        assert isinstance(result["dataset_drift"], bool)

    def test_drift_report_creates_html_file(
        self, reference_parquet, current_df, tmp_path
    ):
        from src.monitoring.drift_detector import run_drift_report
        from pathlib import Path

        run_drift_report(
            reference_path=reference_parquet,
            current_df=current_df,
            output_dir=str(tmp_path),
            report_name="test_report",
        )
        assert (tmp_path / "test_report.html").exists()

    def test_high_drift_detected_on_shifted_data(
        self, reference_parquet, drifted_df, tmp_path
    ):
        from src.monitoring.drift_detector import run_drift_report

        result = run_drift_report(
            reference_path=reference_parquet,
            current_df=drifted_df,
            output_dir=str(tmp_path),
            report_name="drifted_report",
        )
        # A 60-unit shift should be detected as drift
        assert result["drift_score"] > 0.0


class TestAlerting:
    def test_trigger_retrain_called_when_threshold_exceeded(self):
        from src.monitoring.alerting import maybe_trigger_alert

        with patch("src.monitoring.alerting.trigger_retrain_dag") as mock_trigger:
            maybe_trigger_alert(drift_score=0.5, dataset_drift=False)
            mock_trigger.assert_called_once()

    def test_no_trigger_when_below_threshold(self):
        from src.monitoring.alerting import maybe_trigger_alert

        with patch("src.monitoring.alerting.trigger_retrain_dag") as mock_trigger:
            maybe_trigger_alert(drift_score=0.1, dataset_drift=False)
            mock_trigger.assert_not_called()

    def test_trigger_called_when_dataset_drift_true(self):
        from src.monitoring.alerting import maybe_trigger_alert

        with patch("src.monitoring.alerting.trigger_retrain_dag") as mock_trigger:
            maybe_trigger_alert(drift_score=0.05, dataset_drift=True)
            mock_trigger.assert_called_once()
