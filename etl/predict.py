"""
predict.py - forecast stock price metrics.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

HORIZON = 3
MIN_OBS = 3

INDICATORS = [
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "latest_price",
]

_ENV_PATH = Path(__file__).parent.parent / ".env"


def _floor_price(value: float) -> float:
    """Keep price forecasts non-negative."""
    return round(float(max(0.0, value)), 4)


def _prediction_row(
    predicted_year: int,
    predicted_value: float,
    confidence_low: float,
    confidence_high: float,
) -> dict[str, float | int]:
    pred = _floor_price(predicted_value)
    low = _floor_price(confidence_low)
    high = _floor_price(confidence_high)
    return {
        "predicted_year": int(predicted_year),
        "predicted_value": pred,
        "confidence_low": min(low, pred),
        "confidence_high": max(high, pred),
    }


def _get_engine():
    load_dotenv(_ENV_PATH)
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


def _load_indicators(engine) -> pd.DataFrame:
    """
    Read gold.fact_commodity_prices + gold.dim_commodity and return long format.

    Returned id column is named country_id for compatibility with the existing
    evaluation utilities.
    """
    metric_cols = ", ".join(f"f.{c}" for c in INDICATORS)
    query = f"""
        SELECT f.commodity_id,
               d.symbol,
               f.year,
               {metric_cols}
        FROM   gold.fact_commodity_prices f
        JOIN   gold.dim_commodity         d ON d.commodity_id = f.commodity_id
        ORDER  BY d.symbol, f.year
    """
    try:
        wide = pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame(columns=["country_id", "indicator", "year", "value"])

    numeric_cols = [c for c in INDICATORS if c in wide.columns]
    long = wide.melt(
        id_vars=["commodity_id", "symbol", "year"],
        value_vars=numeric_cols,
        var_name="indicator",
        value_name="value",
    )
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.rename(columns={"commodity_id": "country_id"})


def _linear_trend(years: np.ndarray, values: np.ndarray, forecast_years: np.ndarray) -> pd.DataFrame:
    """OLS trend extrapolation with approximate 95% prediction intervals."""
    x = years.astype(float)
    y = values.astype(float)
    n = len(x)

    x_mean = x.mean()
    beta1 = np.sum((x - x_mean) * (y - y.mean())) / np.sum((x - x_mean) ** 2)
    beta0 = y.mean() - beta1 * x_mean

    y_hat = beta0 + beta1 * x
    residuals = y - y_hat
    s2 = np.sum(residuals ** 2) / (n - 2)
    ssx = np.sum((x - x_mean) ** 2)

    rows = []
    for fy in forecast_years:
        pred = beta0 + beta1 * float(fy)
        se_pred = np.sqrt(s2 * (1 + 1 / n + (float(fy) - x_mean) ** 2 / ssx))
        margin = 1.96 * se_pred
        rows.append(_prediction_row(int(fy), pred, pred - margin, pred + margin))
    return pd.DataFrame(rows)


def _holt_smoothing(years: np.ndarray, values: np.ndarray, forecast_years: np.ndarray) -> pd.DataFrame:
    """Holt double exponential smoothing with approximate intervals."""
    y = values.astype(float)
    model = ExponentialSmoothing(y, trend="add", damped_trend=True)
    result = model.fit(optimized=True)

    in_sample_rmse = np.sqrt(np.mean(result.resid ** 2))
    last_year = int(years[-1])
    steps_list = [int(fy) - last_year for fy in forecast_years]
    forecast_vals = result.forecast(max(steps_list))

    rows = []
    for fy, h in zip(forecast_years, steps_list):
        pred = float(forecast_vals[h - 1])
        margin = 1.96 * in_sample_rmse * np.sqrt(h)
        rows.append(_prediction_row(int(fy), pred, pred - margin, pred + margin))
    return pd.DataFrame(rows)


def _fit_series(
    country_id: int,
    indicator: str,
    series: pd.DataFrame,
) -> list[dict]:
    """Fit both models on one ticker metric series."""
    years = series["year"].values
    values = series["value"].values
    last_year = int(years[-1])
    forecast_years = np.array([last_year + h for h in range(1, HORIZON + 1)])

    rows: list[dict] = []
    models = {
        "linear_trend": _linear_trend,
        "holt_smoothing": _holt_smoothing,
    }

    for model_name, fn in models.items():
        try:
            preds = fn(years, values, forecast_years)
            for _, row in preds.iterrows():
                rows.append({
                    "country_id": country_id,
                    "indicator": indicator,
                    "model_name": model_name,
                    "predicted_year": row["predicted_year"],
                    "predicted_value": row["predicted_value"],
                    "confidence_low": row["confidence_low"],
                    "confidence_high": row["confidence_high"],
                })
        except Exception as exc:
            print(f"    [warn] {model_name} failed for indicator={indicator} commodity_id={country_id}: {exc}")

    return rows


def predict(engine=None) -> pd.DataFrame:
    """Fit forecasts for eligible stock metric series and write predictions."""
    if engine is None:
        engine = _get_engine()
        owns_engine = True
    else:
        owns_engine = False

    print("=== Predict - Stock Price Forecasts ===")
    print("  [1/4] Loading stock price data from gold.fact_commodity_prices...")
    df = _load_indicators(engine)
    if df.empty:
        print("  gold.fact_commodity_prices is empty - run load() first. Skipping.")
        return pd.DataFrame()

    print(f"        {len(df):,} rows ({df['country_id'].nunique()} tickers)")

    print("  [2/4] Fitting models...")
    all_rows: list[dict] = []
    for (commodity_id, indicator), grp in df.groupby(["country_id", "indicator"]):
        series = (
            grp[["year", "value"]]
            .dropna(subset=["value"])
            .sort_values("year")
            .drop_duplicates(subset=["year"])
            .reset_index(drop=True)
        )
        if len(series) < MIN_OBS:
            continue
        all_rows.extend(_fit_series(int(commodity_id), indicator, series))

    preds_df = pd.DataFrame(all_rows)
    if preds_df.empty:
        print("  No eligible series found for prediction.")
        return preds_df

    print(f"        {len(preds_df):,} predictions")

    print("  [3/4] Creating gold.fact_predictions...")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.fact_predictions (
                id              SERIAL       PRIMARY KEY,
                commodity_id    INT          NOT NULL REFERENCES gold.dim_commodity (commodity_id),
                indicator       VARCHAR(100) NOT NULL,
                model_name      VARCHAR(100) NOT NULL,
                predicted_year  SMALLINT     NOT NULL,
                predicted_value NUMERIC(18,4),
                confidence_low  NUMERIC(18,4),
                confidence_high NUMERIC(18,4),
                run_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                UNIQUE (commodity_id, indicator, model_name, predicted_year)
            )
        """))
        conn.execute(text("TRUNCATE TABLE gold.fact_predictions RESTART IDENTITY CASCADE"))

    print("  [4/4] Writing predictions...")
    out = preds_df.rename(columns={"country_id": "commodity_id"})
    out.to_sql("fact_predictions", engine, schema="gold", if_exists="append", index=False, chunksize=2_000)
    print(f"        {len(out):,} rows -> gold.fact_predictions")

    if owns_engine:
        engine.dispose()
    return preds_df


if __name__ == "__main__":
    predict()
