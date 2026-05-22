"""
train.py - MLflow-tracked training for stock price forecasting across all timeframes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sqlalchemy import text

# Allow running from project root or from within 04_ml/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from etl.predict import (
    HORIZON,
    MIN_OBS,
    INDICATORS,
    _get_engine,
    _load_indicators,
    _linear_trend,
    _holt_smoothing,
)
from evaluate import compute_metrics, evaluate_all, print_report

_CONFIG_PATH = Path(__file__).parent.parent / "mlflow" / "mlflow_config.yaml"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        return raw.get("mlflow", raw)
    return {
        "tracking_uri":         "mlruns",
        "experiment_name":      "finnhub-stock-forecasting",
        "artifact_location":    "mlartifacts/finnhub-stock-forecasting",
        "registered_model_name":"finnhub-stock-forecaster",
    }


def _resolve_local_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return uri

    path = Path(parsed.path if parsed.scheme == "file" else uri)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _resolve_artifact_location(cfg: dict) -> str:
    raw_location = cfg.get("artifact_location") or "mlartifacts/finnhub-stock-forecasting"
    parsed = urlparse(raw_location)
    if parsed.scheme and parsed.scheme != "file":
        return raw_location

    path = Path(parsed.path if parsed.scheme == "file" else raw_location)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _artifact_location_writable(location: str | None) -> bool:
    if not location:
        return False

    parsed = urlparse(location)
    if parsed.scheme and parsed.scheme != "file":
        return True

    path = Path(parsed.path if parsed.scheme == "file" else location)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(path, os.W_OK)


def _same_artifact_location(left: str | None, right: str) -> bool:
    if not left:
        return False

    def normalise(location: str) -> str:
        parsed = urlparse(location)
        if parsed.scheme and parsed.scheme != "file":
            return location.rstrip("/")
        path = Path(parsed.path if parsed.scheme == "file" else location)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return str(path.resolve(strict=False))

    return normalise(left) == normalise(right)


def _repair_file_store_experiment(cfg: dict) -> None:
    """
    Repair local MLflow metadata created inside Docker.
    """
    tracking_uri = mlflow.get_tracking_uri()
    parsed = urlparse(tracking_uri)
    if parsed.scheme and parsed.scheme != "file":
        return

    tracking_path = Path(parsed.path if parsed.scheme == "file" else tracking_uri)
    if not tracking_path.exists():
        return

    experiment_name = cfg.get("experiment_name", "finnhub-stock-forecasting")
    artifact_location = _resolve_artifact_location(cfg)

    for meta_path in tracking_path.glob("*/meta.yaml"):
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}

        if meta.get("name") != experiment_name or meta.get("lifecycle_stage") == "deleted":
            continue

        current_location = meta.get("artifact_location")
        if (
            _same_artifact_location(current_location, artifact_location)
            and _artifact_location_writable(current_location)
        ):
            return

        meta["artifact_location"] = artifact_location
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        print(f"  MLflow artifact location repaired -> {artifact_location}")
        return


def _configure_mlflow(cfg: dict) -> None:
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        tracking_uri = env_uri
    else:
        tracking_uri = _resolve_local_uri(cfg.get("tracking_uri", "mlruns"))
        
    artifact_location = _resolve_artifact_location(cfg)
    mlflow.set_tracking_uri(tracking_uri)
    
    if not env_uri or env_uri.startswith("file:"):
        _repair_file_store_experiment(cfg)

    client = MlflowClient()
    experiment_name = cfg.get("experiment_name", "finnhub-stock-forecasting")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)

    mlflow.set_experiment(experiment_name)


def _fit_model(
    df: pd.DataFrame,
    model_name: str,
    timeframe: str,
    test_size: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit one model across every (ticker, indicator) group in df for the given timeframe.
    """
    fn = _linear_trend if model_name == "linear_trend" else _holt_smoothing

    pred_rows: list[dict]   = []
    metric_rows: list[dict] = []

    for (commodity_id, indicator), grp in df.groupby(["country_id", "indicator"]):
        series = (
            grp[["time_index", "value"]]
            .dropna(subset=["value"])
            .sort_values("time_index")
            .drop_duplicates(subset=["time_index"])
            .reset_index(drop=True)
        )

        if len(series) < MIN_OBS:
            continue

        # Hold-out split: keep last `test_size` points for evaluation
        if len(series) > test_size:
            train = series.iloc[:-test_size]
            test  = series.iloc[-test_size:]
        else:
            train = series
            test  = None

        dates = train["time_index"].values
        values = train["value"].values
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

        try:
            preds = fn(x, values, forecast_steps, forecast_dates)
        except Exception as exc:
            print(f"    [warn] {model_name} failed commodity_id={commodity_id} {indicator} timeframe={timeframe}: {exc}")
            continue

        for _, row in preds.iterrows():
            pred_rows.append({
                "commodity_id":    int(commodity_id),
                "timeframe":       timeframe,
                "indicator":       indicator,
                "model_name":      model_name,
                "predicted_time":  row["predicted_time"],
                "predicted_value": row["predicted_value"],
                "confidence_low":  row["confidence_low"],
                "confidence_high": row["confidence_high"],
            })

        # Hold-out metrics
        if test is not None:
            for _, trow in test.iterrows():
                tdat   = trow["time_index"]
                pred = preds.loc[preds["predicted_time"] == tdat, "predicted_value"]
                if pred.empty:
                    continue
                m = compute_metrics(
                    np.array([trow["value"]]),
                    pred.values[:1],
                )
                metric_rows.append({
                    "commodity_id":  int(commodity_id),
                    "indicator": indicator,
                    **m,
                })

    pred_df   = pd.DataFrame(pred_rows)
    metric_df = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame(
        columns=["commodity_id", "indicator", "mae", "rmse", "mape", "r2"]
    )
    return pred_df, metric_df


def _run_experiment(
    df: pd.DataFrame,
    model_name: str,
    timeframe: str,
    cfg: dict,
    test_size: int,
    register: bool,
) -> pd.DataFrame:
    """Open one MLflow run, fit the model, log everything, return predictions."""
    run_name = f"{model_name}_{timeframe}"
    with mlflow.start_run(run_name=run_name) as run:
        # ── Parameters ──────────────────────────────────────────────────────
        mlflow.log_params({
            "model_name":  model_name,
            "timeframe":   timeframe,
            "horizon":     HORIZON,
            "min_obs":     MIN_OBS,
            "test_size":   test_size,
            "n_tickers":    int(df["country_id"].nunique()),
            "n_indicators":len(INDICATORS),
            "indicators":  ",".join(INDICATORS),
        })
        mlflow.set_tags({
            "project": cfg.get("default_tags", {}).get("project", "finnhub-trading-pipeline"),
            "team":    cfg.get("default_tags", {}).get("team", "data-engineering"),
            "timeframe": timeframe,
        })

        # ── Fit ─────────────────────────────────────────────────────────────
        preds_df, metric_df = _fit_model(df, model_name, timeframe=timeframe, test_size=test_size)
        n_series = int(df.groupby(["country_id", "indicator"]).ngroups)
        n_fitted = int(preds_df["commodity_id"].nunique()) if not preds_df.empty else 0

        mlflow.log_metric("n_series_total",  n_series)
        mlflow.log_metric("n_series_fitted", n_fitted)
        mlflow.log_metric("n_predictions",   len(preds_df))

        # ── Metrics ─────────────────────────────────────────────────────────
        if not metric_df.empty:
            mlflow.log_metric("avg_mae",  round(float(metric_df["mae"].mean()),  6))
            mlflow.log_metric("avg_rmse", round(float(metric_df["rmse"].mean()), 6))
            mlflow.log_metric("avg_mape", round(float(metric_df["mape"].mean()), 6))
            mlflow.log_metric("avg_r2",   round(float(metric_df["r2"].mean()),   6))
            print(f"  {model_name}: MAE={metric_df['mae'].mean():.4f}  "
                  f"RMSE={metric_df['rmse'].mean():.4f}  "
                  f"R²={metric_df['r2'].mean():.4f}  "
                  f"({len(metric_df)} eval points)")
        else:
            mlflow.log_metric("avg_mae", -1)
            print(f"  {model_name}: metrics not computed "
                  f"(need >{MIN_OBS + test_size} historical data points to evaluate)")

        # ── Artifacts ───────────────────────────────────────────────────────
        if not preds_df.empty:
            with tempfile.TemporaryDirectory() as tmp:
                csv_path = os.path.join(tmp, f"predictions_{model_name}_{timeframe}.csv")
                preds_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path, artifact_path="predictions")

            if not metric_df.empty:
                with tempfile.TemporaryDirectory() as tmp:
                    m_path = os.path.join(tmp, f"metrics_{model_name}_{timeframe}.csv")
                    metric_df.to_csv(m_path, index=False)
                    mlflow.log_artifact(m_path, artifact_path="metrics")

        print(f"  MLflow run: {run.info.run_id}  ({len(preds_df):,} predictions logged)")

    return preds_df


def _write_predictions(all_preds: list[pd.DataFrame], engine) -> None:
    if not all_preds:
        return
    combined = pd.concat([p for p in all_preds if not p.empty], ignore_index=True)
    if combined.empty:
        print("  No predictions to write.")
        return

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

    combined.to_sql(
        "fact_predictions", engine, schema="gold",
        if_exists="append", index=False, chunksize=2_000,
    )
    print(f"  {len(combined):,} rows -> gold.fact_predictions")


def train(
    register: bool = True,
    filter_indicator: str | None = None,
) -> None:
    cfg = _load_config()

    _configure_mlflow(cfg)

    test_size = int(cfg.get("training", {}).get("test_size", 1))

    engine = _get_engine()

    print("=== Step 4 — ML Training (MLflow) ===")
    print(f"  Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"  Experiment   : {cfg.get('experiment_name')}")

    print("\n  Loading data from gold.fact_commodity_prices...")
    df = _load_indicators(engine)

    if df.empty:
        print("  gold.fact_commodity_prices is empty - run main.py first, then re-run train.py.")
        engine.dispose()
        return

    if filter_indicator:
        df = df[df["indicator"] == filter_indicator]

    print(f"  {len(df):,} rows | {df['country_id'].nunique()} tickers "
          f"| {df['indicator'].nunique()} indicators\n")

    model_types = ["linear_trend", "holt_smoothing"]
    all_preds: list[pd.DataFrame] = []

    # Loop over all active timeframes in the data
    timeframes = df["timeframe"].unique() if "timeframe" in df.columns else ["year"]
    for timeframe in timeframes:
        print(f"\n==================================================")
        print(f"Timeframe: {timeframe.upper()}")
        print(f"==================================================")
        tf_df = df[df["timeframe"] == timeframe]
        if tf_df.empty:
            continue

        for model_name in model_types:
            run_name = f"{model_name}_{timeframe}"
            print(f"--- {run_name} ---")
            preds = _run_experiment(tf_df, model_name, timeframe, cfg, test_size, register)
            all_preds.append(preds)

    print("\n=== Writing gold.fact_predictions ===")
    _write_predictions(all_preds, engine)

    engine.dispose()

    print(f"\n  Done. View results:")
    print(f"    mlflow ui --port 5001")
    print(f"    open http://localhost:5001")


if __name__ == "__main__":
    load_dotenv(_PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="MLflow-tracked stock price forecasting")
    parser.add_argument("--no-register",  action="store_true", help="skip model registry")
    parser.add_argument("--indicator",    default=None,        help="train one indicator only")
    args = parser.parse_args()
    train(register=not args.no_register, filter_indicator=args.indicator)
