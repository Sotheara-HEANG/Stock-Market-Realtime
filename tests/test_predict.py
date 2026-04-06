"""
test_predict.py — unit tests for the two forecasting model functions.

Tests use synthetic time series so no database or CSV files are required.
"""

import numpy as np
import pandas as pd
import pytest

from etl.predict import HORIZON, _holt_smoothing, _linear_trend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_series():
    """Clean upward linear trend: value = 2*t + 5 for t in 0..14."""
    rng = np.random.default_rng(42)
    years = np.arange(2000, 2015)
    values = 2.0 * (years - 2000) + 5.0 + rng.normal(0, 0.1, len(years))
    forecast_years = np.arange(2015, 2015 + HORIZON)
    return years, values, forecast_years


@pytest.fixture
def flat_series():
    """Flat series with minor noise — tests near-zero trend."""
    rng = np.random.default_rng(0)
    years = np.arange(2005, 2020)
    values = 3.0 + rng.normal(0, 0.05, len(years))
    forecast_years = np.arange(2020, 2020 + HORIZON)
    return years, values, forecast_years


# ---------------------------------------------------------------------------
# _linear_trend
# ---------------------------------------------------------------------------

class TestLinearTrend:
    def test_output_shape(self, linear_series):
        years, values, forecast_years = linear_series
        result = _linear_trend(years, values, forecast_years)
        assert len(result) == HORIZON

    def test_output_columns(self, linear_series):
        years, values, forecast_years = linear_series
        result = _linear_trend(years, values, forecast_years)
        assert list(result.columns) == [
            "predicted_year", "predicted_value", "confidence_low", "confidence_high"
        ]

    def test_forecast_years_match(self, linear_series):
        years, values, forecast_years = linear_series
        result = _linear_trend(years, values, forecast_years)
        assert list(result["predicted_year"]) == list(forecast_years)

    def test_confidence_interval_ordering(self, linear_series):
        years, values, forecast_years = linear_series
        result = _linear_trend(years, values, forecast_years)
        assert (result["confidence_low"] < result["predicted_value"]).all()
        assert (result["predicted_value"] < result["confidence_high"]).all()

    def test_upward_trend_predicts_increasing(self, linear_series):
        years, values, forecast_years = linear_series
        result = _linear_trend(years, values, forecast_years)
        predicted = result["predicted_value"].values
        assert predicted[-1] > predicted[0], "Upward trend should produce increasing forecasts"

    def test_flat_series_predicts_stable(self, flat_series):
        years, values, forecast_years = flat_series
        result = _linear_trend(years, values, forecast_years)
        spread = result["predicted_value"].max() - result["predicted_value"].min()
        assert spread < 1.0, f"Flat series should have stable forecasts, spread={spread:.3f}"

    def test_no_nan_in_output(self, linear_series):
        years, values, forecast_years = linear_series
        result = _linear_trend(years, values, forecast_years)
        assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# _holt_smoothing
# ---------------------------------------------------------------------------

class TestHoltSmoothing:
    def test_output_shape(self, linear_series):
        years, values, forecast_years = linear_series
        result = _holt_smoothing(years, values, forecast_years)
        assert len(result) == HORIZON

    def test_output_columns(self, linear_series):
        years, values, forecast_years = linear_series
        result = _holt_smoothing(years, values, forecast_years)
        assert list(result.columns) == [
            "predicted_year", "predicted_value", "confidence_low", "confidence_high"
        ]

    def test_forecast_years_match(self, linear_series):
        years, values, forecast_years = linear_series
        result = _holt_smoothing(years, values, forecast_years)
        assert list(result["predicted_year"]) == list(forecast_years)

    def test_confidence_interval_ordering(self, linear_series):
        years, values, forecast_years = linear_series
        result = _holt_smoothing(years, values, forecast_years)
        assert (result["confidence_low"] < result["predicted_value"]).all()
        assert (result["predicted_value"] < result["confidence_high"]).all()

    def test_intervals_widen_further_ahead(self, linear_series):
        """Confidence bands should widen (or stay equal) as h increases."""
        years, values, forecast_years = linear_series
        result = _holt_smoothing(years, values, forecast_years)
        widths = (result["confidence_high"] - result["confidence_low"]).values
        assert widths[-1] >= widths[0], "Intervals should widen for later forecast horizons"

    def test_no_nan_in_output(self, linear_series):
        years, values, forecast_years = linear_series
        result = _holt_smoothing(years, values, forecast_years)
        assert not result.isnull().any().any()
