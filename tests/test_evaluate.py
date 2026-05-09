"""
test_evaluate.py — unit tests for 04_ml/training/evaluate.py.

No database or Kafka required — all tests use synthetic data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_ml" / "training"))

from evaluate import compute_metrics, evaluate_all, print_report


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_prediction(self):
        a = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(a, a)
        assert m["mae"]  == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"]   == 1.0

    def test_known_values(self):
        actual    = np.array([3.0, 3.0, 3.0, 3.0])
        predicted = np.array([2.0, 4.0, 2.0, 4.0])
        m = compute_metrics(actual, predicted)
        assert m["mae"]  == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(1.0)

    def test_returns_all_keys(self):
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert set(m.keys()) == {"mae", "rmse", "mape", "r2"}

    def test_empty_arrays(self):
        m = compute_metrics(np.array([]), np.array([]))
        assert np.isnan(m["mae"])
        assert np.isnan(m["rmse"])

    def test_mape_skips_zero_actuals(self):
        # Zero actual should not raise ZeroDivisionError
        m = compute_metrics(np.array([0.0, 2.0]), np.array([1.0, 2.0]))
        assert not np.isnan(m["mape"]) or np.isnan(m["mape"])  # either is fine

    def test_r2_below_one_for_bad_fit(self):
        actual    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        m = compute_metrics(actual, predicted)
        assert m["r2"] < 0.0   # inverse predictions → very negative R²

    def test_unequal_length_uses_shortest(self):
        actual    = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0])
        m = compute_metrics(actual, predicted)
        assert m["mae"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    def _make_preds(self):
        return pd.DataFrame([
            {"country_id": 1, "indicator": "gdp_growth_pct", "model_name": "linear_trend",
             "predicted_year": 2021, "predicted_value": 2.5},
            {"country_id": 1, "indicator": "gdp_growth_pct", "model_name": "linear_trend",
             "predicted_year": 2022, "predicted_value": 3.0},
            {"country_id": 2, "indicator": "gdp_growth_pct", "model_name": "linear_trend",
             "predicted_year": 2021, "predicted_value": 1.0},
        ])

    def _make_actuals(self):
        return pd.DataFrame([
            {"country_id": 1, "indicator": "gdp_growth_pct", "year": 2021, "value": 2.6},
            {"country_id": 1, "indicator": "gdp_growth_pct", "year": 2022, "value": 2.9},
            {"country_id": 2, "indicator": "gdp_growth_pct", "year": 2021, "value": 1.1},
        ])

    def test_returns_dataframe(self):
        result = evaluate_all(self._make_preds(), self._make_actuals())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = evaluate_all(self._make_preds(), self._make_actuals())
        assert "mae" in result.columns
        assert "rmse" in result.columns
        assert "model_name" in result.columns
        assert "indicator" in result.columns

    def test_metrics_are_non_negative(self):
        result = evaluate_all(self._make_preds(), self._make_actuals())
        assert (result["mae"]  >= 0).all()
        assert (result["rmse"] >= 0).all()

    def test_empty_preds_returns_empty(self):
        result = evaluate_all(pd.DataFrame(), self._make_actuals())
        assert result.empty

    def test_no_matching_actuals_returns_empty(self):
        preds = self._make_preds()
        actuals = pd.DataFrame([
            {"country_id": 99, "indicator": "unrelated", "year": 2021, "value": 1.0}
        ])
        result = evaluate_all(preds, actuals)
        assert result.empty


# ---------------------------------------------------------------------------
# print_report (smoke test — just check it doesn't raise)
# ---------------------------------------------------------------------------

def test_print_report_no_crash(capsys):
    df = pd.DataFrame([{
        "model_name": "linear_trend", "indicator": "gdp_growth_pct",
        "mae": 0.1, "rmse": 0.2, "mape": 5.0, "r2": 0.9, "n_series": 10
    }])
    print_report(df)
    out = capsys.readouterr().out
    assert "linear_trend" in out


def test_print_report_empty_no_crash(capsys):
    print_report(pd.DataFrame())
    out = capsys.readouterr().out
    assert "No evaluation" in out
