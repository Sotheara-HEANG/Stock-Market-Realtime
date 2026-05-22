"""
predict.py - forecast stock price metrics for Days, Weeks, Months, and Years timeframes.
"""

from __future__ import annotations

import datetime as dt
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
    predicted_time: dt.date,
    predicted_value: float,
    confidence_low: float,
    confidence_high: float,
) -> dict[str, float | dt.date]:
    pred = _floor_price(predicted_value)
    low = _floor_price(confidence_low)
    high = _floor_price(confidence_high)
    return {
        "predicted_time": predicted_time,
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
    """
    metric_cols = ", ".join(f"f.{c}" for c in INDICATORS)
    query = f"""
        SELECT f.commodity_id,
               d.symbol,
               f.timeframe,
               f.time_index,
               {metric_cols}
        FROM   gold.fact_commodity_prices f
        JOIN   gold.dim_commodity         d ON d.commodity_id = f.commodity_id
        ORDER  BY d.symbol, f.timeframe, f.time_index
    """
    try:
        wide = pd.read_sql(query, engine)
    except Exception as exc:
        print(f"  [error] Loading indicators failed: {exc}")
        return pd.DataFrame(columns=["country_id", "indicator", "timeframe", "time_index", "value"])

    numeric_cols = [c for c in INDICATORS if c in wide.columns]
    long = wide.melt(
        id_vars=["commodity_id", "symbol", "timeframe", "time_index"],
        value_vars=numeric_cols,
        var_name="indicator",
        value_name="value",
    )
    long["time_index"] = pd.to_datetime(long["time_index"]).dt.date
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.rename(columns={"commodity_id": "country_id"})


def _linear_trend(
    x: np.ndarray,
    y: np.ndarray,
    forecast_steps: np.ndarray,
    forecast_dates: list[dt.date],
) -> pd.DataFrame:
    """OLS trend extrapolation with approximate 95% prediction intervals."""
    n = len(x)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        beta1 = 0.0
    else:
        beta1 = np.sum((x - x_mean) * (y - y.mean())) / denom
    beta0 = y.mean() - beta1 * x_mean

    y_hat = beta0 + beta1 * x
    residuals = y - y_hat
    s2 = np.sum(residuals ** 2) / (n - 2) if n > 2 else 0.0001
    if s2 <= 0:
        s2 = 0.0001
    ssx = denom if denom > 0 else 0.0001

    rows = []
    for step, fdate in zip(forecast_steps, forecast_dates):
        pred = beta0 + beta1 * float(step)
        se_pred = np.sqrt(s2 * (1 + 1 / n + (float(step) - x_mean) ** 2 / ssx))
        margin = 1.96 * se_pred
        rows.append(_prediction_row(fdate, pred, pred - margin, pred + margin))
    return pd.DataFrame(rows)


def _holt_smoothing(
    x: np.ndarray,
    y: np.ndarray,
    forecast_steps: np.ndarray,
    forecast_dates: list[dt.date],
) -> pd.DataFrame:
    """Holt double exponential smoothing with approximate intervals."""
    model = ExponentialSmoothing(y, trend="add", damped_trend=True)
    result = model.fit(optimized=True)

    in_sample_rmse = np.sqrt(np.mean(result.resid ** 2))
    if in_sample_rmse <= 0:
        in_sample_rmse = 0.01
    forecast_vals = result.forecast(HORIZON)

    rows = []
    for h, fdate in enumerate(forecast_dates):
        pred = float(forecast_vals[h])
        margin = 1.96 * in_sample_rmse * np.sqrt(h + 1)
        rows.append(_prediction_row(fdate, pred, pred - margin, pred + margin))
    return pd.DataFrame(rows)


def _fit_series(
    country_id: int,
    timeframe: str,
    indicator: str,
    series: pd.DataFrame,
) -> list[dict]:
    """Fit both models on one ticker metric series."""
    dates = series["time_index"].values
    values = series["value"].values
    n = len(dates)
    
    x = np.arange(n)
    forecast_steps = np.arange(n, n + HORIZON)
    
    last_date = dates[-1]
    forecast_dates = []
    for step in forecast_steps:
        offset = int(step - (n - 1))
        if timeframe == 'day':
            fdate = last_date + dt.timedelta(days=offset)
        elif timeframe == 'week':
            fdate = last_date + dt.timedelta(weeks=offset)
        elif timeframe == 'month':
            year = last_date.year + (last_date.month - 1 + offset) // 12
            month = (last_date.month - 1 + offset) % 12 + 1
            fdate = dt.date(year, month, 1)
        elif timeframe == 'year':
            fdate = dt.date(last_date.year + offset, 1, 1)
        else:
            fdate = last_date + dt.timedelta(days=offset)
        forecast_dates.append(fdate)

    rows: list[dict] = []
    models = {
        "linear_trend": _linear_trend,
        "holt_smoothing": _holt_smoothing,
    }

    for model_name, fn in models.items():
        try:
            preds = fn(x, values, forecast_steps, forecast_dates)
            for _, row in preds.iterrows():
                rows.append({
                    "country_id": country_id,
                    "timeframe": timeframe,
                    "indicator": indicator,
                    "model_name": model_name,
                    "predicted_time": row["predicted_time"],
                    "predicted_value": row["predicted_value"],
                    "confidence_low": row["confidence_low"],
                    "confidence_high": row["confidence_high"],
                })
        except Exception as exc:
            print(f"    [warn] {model_name} failed for indicator={indicator} commodity_id={country_id} timeframe={timeframe}: {exc}")

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
    for (commodity_id, timeframe, indicator), grp in df.groupby(["country_id", "timeframe", "indicator"]):
        series = (
            grp[["time_index", "value"]]
            .dropna(subset=["value"])
            .sort_values("time_index")
            .drop_duplicates(subset=["time_index"])
            .reset_index(drop=True)
        )
        if len(series) < MIN_OBS:
            continue
        all_rows.extend(_fit_series(int(commodity_id), timeframe, indicator, series))

    preds_df = pd.DataFrame(all_rows)
    if preds_df.empty:
        print("  No eligible series found for prediction.")
        return preds_df

    print(f"        {len(preds_df):,} predictions")

    print("  [3/4] Creating gold.fact_predictions...")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
        conn.execute(text("DROP TABLE IF EXISTS gold.fact_predictions CASCADE"))
        conn.execute(text("""
            CREATE TABLE gold.fact_predictions (
                id              SERIAL       PRIMARY KEY,
                commodity_id    INT          NOT NULL REFERENCES gold.dim_commodity (commodity_id),
                timeframe       VARCHAR(10)  NOT NULL,
                indicator       VARCHAR(100) NOT NULL,
                model_name      VARCHAR(100) NOT NULL,
                predicted_time  DATE         NOT NULL,
                predicted_value NUMERIC(18,4),
                confidence_low  NUMERIC(18,4),
                confidence_high NUMERIC(18,4),
                run_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                UNIQUE (commodity_id, timeframe, indicator, model_name, predicted_time)
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gold_pred_commodity ON gold.fact_predictions (commodity_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gold_pred_model ON gold.fact_predictions (model_name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gold_pred_tf_time ON gold.fact_predictions (timeframe, predicted_time)"))

    print("  [4/4] Writing predictions...")
    out = preds_df.rename(columns={"country_id": "commodity_id"})
    out.to_sql("fact_predictions", engine, schema="gold", if_exists="append", index=False, chunksize=2_000)
    print(f"        {len(out):,} rows -> gold.fact_predictions")

    if owns_engine:
        engine.dispose()
    return preds_df


if __name__ == "__main__":
    predict()
