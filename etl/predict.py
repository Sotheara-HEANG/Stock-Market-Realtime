"""
predict.py — forecast economic and governance indicators per country.

Two models per (country, indicator) series:
  1. linear_trend     — OLS trend extrapolation with 95% prediction intervals
  2. holt_smoothing   — Holt double exponential smoothing (handles trend)

Both are fitted on historical data up to last_data_year and produce
predictions for last_data_year+1 through last_data_year+HORIZON years.

Minimum 10 non-null observations required for a series to be modelled.

Indicators modelled:
  - gdp_growth_pct
  - inflation_pct
  - unemployment_pct
  - governance_composite
  - hdi_value

Output: writes to the `predictions` table in PostgreSQL.

Usage:
    from etl.predict import predict
    predict(engine)               # reads indicators table, writes predictions

    # or run standalone:
    python -m etl.predict
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZON = 3          # periods ahead to forecast
MIN_OBS = 3          # minimum non-null data points to fit a model

INDICATORS = [
    "current_price_usd",
    "price_change_pct",
    "trading_volume",
    "intraday_range_pct",
]

_ENV_PATH = Path(__file__).parent.parent / ".env"


# ---------------------------------------------------------------------------
# DB helpers (shared pattern with load.py)
# ---------------------------------------------------------------------------

def _get_engine():
    load_dotenv(_ENV_PATH)
    host     = os.environ["DB_HOST"]
    port     = os.environ.get("DB_PORT", "5432")
    dbname   = os.environ["DB_NAME"]
    user     = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


def _load_indicators(engine) -> pd.DataFrame:
    """
    Read gold.fact_prices + gold.dim_asset and return a long-format DataFrame:
        asset_id, indicator, year, value
    melted from the price columns, filtered to INDICATORS.
    """
    price_cols = ", ".join(f"f.{c}" for c in INDICATORS if c != "price_momentum")
    query = f"""
        SELECT f.asset_id,
               d.company_name AS asset_name,
               f.year,
               {price_cols}
        FROM   gold.fact_prices f
        JOIN   gold.dim_asset   d ON d.asset_id = f.asset_id
        ORDER  BY d.company_name, f.year
    """
    try:
        wide = pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame(columns=["asset_id", "indicator", "year", "value"])

    numeric_cols = [c for c in INDICATORS if c in wide.columns and c != "price_momentum"]
    long = wide.melt(
        id_vars=["asset_id", "asset_name", "year"],
        value_vars=numeric_cols,
        var_name="indicator",
        value_name="value",
    )
    long["year"]  = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.rename(columns={"asset_id": "country_id"})


# ---------------------------------------------------------------------------
# Model 1: Linear trend with 95% prediction intervals
# ---------------------------------------------------------------------------

def _linear_trend(years: np.ndarray, values: np.ndarray, forecast_years: np.ndarray
                  ) -> pd.DataFrame:
    """
    OLS: value ~ year.  Returns a DataFrame with columns:
        predicted_year, predicted_value, confidence_low, confidence_high
    Confidence interval = ±1.96 * SE of prediction (not just SE of fit).
    """
    x = years.astype(float)
    y = values.astype(float)
    n = len(x)

    # Fit via normal equations
    x_mean = x.mean()
    beta1  = np.sum((x - x_mean) * (y - y.mean())) / np.sum((x - x_mean) ** 2)
    beta0  = y.mean() - beta1 * x_mean

    y_hat    = beta0 + beta1 * x
    residuals = y - y_hat
    s2       = np.sum(residuals ** 2) / (n - 2)   # MSE
    ssx      = np.sum((x - x_mean) ** 2)

    rows = []
    for fy in forecast_years:
        pred = beta0 + beta1 * float(fy)
        # SE of prediction (includes variance of the new observation)
        se_pred = np.sqrt(s2 * (1 + 1 / n + (float(fy) - x_mean) ** 2 / ssx))
        margin  = 1.96 * se_pred
        rows.append({
            "predicted_year":  int(fy),
            "predicted_value": round(float(pred), 4),
            "confidence_low":  round(float(pred - margin), 4),
            "confidence_high": round(float(pred + margin), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model 2: Holt double exponential smoothing
# ---------------------------------------------------------------------------

def _holt_smoothing(years: np.ndarray, values: np.ndarray, forecast_years: np.ndarray
                    ) -> pd.DataFrame:
    """
    Holt's linear (double) exponential smoothing.
    Confidence interval approximated as ±1.96 * in-sample RMSE * sqrt(h),
    where h = steps ahead (a common heuristic for ETS models).
    """
    y = values.astype(float)

    model  = ExponentialSmoothing(y, trend="add", damped_trend=True)
    result = model.fit(optimized=True)

    in_sample_rmse = np.sqrt(np.mean(result.resid ** 2))
    last_year      = int(years[-1])
    steps_list     = [int(fy) - last_year for fy in forecast_years]
    forecast_vals  = result.forecast(max(steps_list))

    rows = []
    for fy, h in zip(forecast_years, steps_list):
        pred   = float(forecast_vals[h - 1])
        margin = 1.96 * in_sample_rmse * np.sqrt(h)
        rows.append({
            "predicted_year":  int(fy),
            "predicted_value": round(pred, 4),
            "confidence_low":  round(pred - margin, 4),
            "confidence_high": round(pred + margin, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core: fit both models for one (country, indicator) series
# ---------------------------------------------------------------------------

def _fit_series(
    country_id: int,
    indicator:  str,
    series:     pd.DataFrame,   # columns: year, value  (sorted, no NaNs)
) -> list[dict]:
    """
    Fit linear_trend and holt_smoothing on one series.
    Returns a list of prediction dicts ready to go into the predictions table.
    """
    years  = series["year"].values
    values = series["value"].values
    last_year     = int(years[-1])
    forecast_years = np.array([last_year + h for h in range(1, HORIZON + 1)])

    rows: list[dict] = []
    models = {
        "linear_trend":   _linear_trend,
        "holt_smoothing": _holt_smoothing,
    }

    for model_name, fn in models.items():
        try:
            preds = fn(years, values, forecast_years)
            for _, row in preds.iterrows():
                rows.append({
                    "country_id":      country_id,
                    "indicator":       indicator,
                    "model_name":      model_name,
                    "predicted_year":  row["predicted_year"],
                    "predicted_value": row["predicted_value"],
                    "confidence_low":  row["confidence_low"],
                    "confidence_high": row["confidence_high"],
                })
        except Exception as exc:
            # Log but don't abort the whole run
            print(f"    [warn] {model_name} failed for indicator={indicator} "
                  f"country_id={country_id}: {exc}")

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def predict(engine=None) -> pd.DataFrame:
    """
    Fit models on every eligible (country, indicator) series, then write
    results to the `predictions` table.

    Args:
        engine: SQLAlchemy engine.  If None, builds one from .env.

    Returns:
        DataFrame of all predictions written (for inspection / testing).
    """
    if engine is None:
        engine = _get_engine()

    print("=== Predict ===")

    print("  [1/4] Loading price data from gold.fact_prices...")
    df = _load_indicators(engine)
    if df.empty:
        print("  gold.fact_prices is empty — run load() first. Skipping.")
        return pd.DataFrame()
    print(f"        {len(df):,} rows  ({df['indicator'].nunique()} indicators, "
          f"{df['country_id'].nunique()} assets)")

    print(f"  [2/4] Fitting models (horizon={HORIZON} yrs, min_obs={MIN_OBS})...")
    all_rows: list[dict] = []
    skipped = 0

    for (asset_id, indicator), grp in df.groupby(["country_id", "indicator"]):
        series = (
            grp[["year", "value"]]
            .dropna(subset=["value"])
            .sort_values("year")
            .drop_duplicates(subset=["year"])
            .reset_index(drop=True)
        )

        if len(series) < MIN_OBS:
            skipped += 1
            continue

        all_rows.extend(_fit_series(int(asset_id), indicator, series))

    total_series  = df.groupby(["country_id", "indicator"]).ngroups
    fitted_series = total_series - skipped
    print(f"        {fitted_series:,} series fitted  ({skipped:,} skipped — fewer than {MIN_OBS} obs)")
    print(f"        {len(all_rows):,} prediction rows generated")

    if not all_rows:
        print("  No predictions to write (not enough historical data yet).")
        return pd.DataFrame()

    preds_df = pd.DataFrame(all_rows)

    print("  [3/4] Writing to gold.fact_predictions...")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.fact_predictions (
                id              SERIAL       PRIMARY KEY,
                asset_id        INT          NOT NULL,
                indicator       VARCHAR(100) NOT NULL,
                model_name      VARCHAR(100) NOT NULL,
                predicted_year  SMALLINT     NOT NULL,
                predicted_value NUMERIC(18,4),
                confidence_low  NUMERIC(18,4),
                confidence_high NUMERIC(18,4),
                run_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                UNIQUE (asset_id, indicator, model_name, predicted_year)
            )
        """))
        conn.execute(text("TRUNCATE TABLE gold.fact_predictions RESTART IDENTITY CASCADE"))

    preds_df = preds_df.rename(columns={"country_id": "asset_id"})
    preds_df.to_sql(
        "fact_predictions", engine, schema="gold",
        if_exists="append", index=False, chunksize=2_000,
    )
    print(f"        {len(preds_df):,} rows → gold.fact_predictions")

    print("  [4/4] Summary:")
    summary = (
        preds_df.groupby(["model_name", "indicator"])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["model_name", "indicator"])
    )
    print(summary.to_string(index=False))

    engine.dispose()
    print("\n  Done.")
    return preds_df


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    predict()
