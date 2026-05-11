"""
train.py — MLflow-tracked training for finance price forecasting.

For each model type (linear_trend, holt_smoothing):
  1. Opens an MLflow run
  2. Logs hyperparameters (horizon, min_obs, n_assets, indicators)
  3. Fits models on gold.fact_prices data
  4. Logs metrics (MAE, RMSE, MAPE, R²) when hold-out data is available
  5. Saves a predictions CSV as an MLflow artifact
  6. Writes all predictions to gold.fact_predictions

MLflow UI:
    mlflow ui --port 5001
    # then open http://localhost:5001

Usage:
    python 04_ml/training/train.py
    python 04_ml/training/train.py --no-register
    python 04_ml/training/train.py --indicator current_price_usd
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        return raw.get("mlflow", raw)
    return {
        "tracking_uri":         "mlruns",
        "experiment_name":      "finance-price-forecasting",
        "registered_model_name":"finance-forecaster",
    }


# ---------------------------------------------------------------------------
# Fitting one model type across all (asset, indicator) series
# ---------------------------------------------------------------------------

def _fit_model(
    df: pd.DataFrame,
    model_name: str,
    test_size: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit one model across every (asset_id, indicator) group in df.

    Returns:
        predictions_df — ready to write to gold.fact_predictions
        metrics_df     — per-group hold-out metrics (empty if not enough data)
    """
    fn = _linear_trend if model_name == "linear_trend" else _holt_smoothing

    pred_rows: list[dict]   = []
    metric_rows: list[dict] = []

    for (asset_id, indicator), grp in df.groupby(["country_id", "indicator"]):
        series = (
            grp[["year", "value"]]
            .dropna(subset=["value"])
            .sort_values("year")
            .drop_duplicates(subset=["year"])
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

        last_year      = int(train["year"].values[-1])
        forecast_years = np.array([last_year + h for h in range(1, HORIZON + 1)])

        try:
            preds = fn(train["year"].values, train["value"].values, forecast_years)
        except Exception as exc:
            print(f"    [warn] {model_name} failed asset={asset_id} {indicator}: {exc}")
            continue

        for _, row in preds.iterrows():
            pred_rows.append({
                "asset_id":        int(asset_id),
                "indicator":       indicator,
                "model_name":      model_name,
                "predicted_year":  row["predicted_year"],
                "predicted_value": row["predicted_value"],
                "confidence_low":  row["confidence_low"],
                "confidence_high": row["confidence_high"],
            })

        # Hold-out metrics
        if test is not None:
            for _, trow in test.iterrows():
                ty   = int(trow["year"])
                pred = preds.loc[preds["predicted_year"] == ty, "predicted_value"]
                if pred.empty:
                    continue
                m = compute_metrics(
                    np.array([trow["value"]]),
                    pred.values[:1],
                )
                metric_rows.append({
                    "asset_id":  int(asset_id),
                    "indicator": indicator,
                    **m,
                })

    pred_df   = pd.DataFrame(pred_rows)
    metric_df = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame(
        columns=["asset_id", "indicator", "mae", "rmse", "mape", "r2"]
    )
    return pred_df, metric_df


# ---------------------------------------------------------------------------
# MLflow run for one model type
# ---------------------------------------------------------------------------

def _run_experiment(
    df: pd.DataFrame,
    model_name: str,
    cfg: dict,
    test_size: int,
    register: bool,
) -> pd.DataFrame:
    """Open one MLflow run, fit the model, log everything, return predictions."""
    with mlflow.start_run(run_name=model_name) as run:
        # ── Parameters ──────────────────────────────────────────────────────
        mlflow.log_params({
            "model_name":  model_name,
            "horizon":     HORIZON,
            "min_obs":     MIN_OBS,
            "test_size":   test_size,
            "n_assets":    int(df["country_id"].nunique()),
            "n_indicators":len(INDICATORS),
            "indicators":  ",".join(INDICATORS),
        })
        mlflow.set_tags({
            "project": cfg.get("default_tags", {}).get("project", "finance-pipeline"),
            "team":    cfg.get("default_tags", {}).get("team", "data-engineering"),
        })

        # ── Fit ─────────────────────────────────────────────────────────────
        preds_df, metric_df = _fit_model(df, model_name, test_size=test_size)
        n_series = int(df.groupby(["country_id", "indicator"]).ngroups)
        n_fitted = int(preds_df["asset_id"].nunique()) if not preds_df.empty else 0

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
                  f"(need >{MIN_OBS + test_size} pipeline runs to accumulate history)")

        # ── Artifacts ───────────────────────────────────────────────────────
        if not preds_df.empty:
            with tempfile.TemporaryDirectory() as tmp:
                csv_path = os.path.join(tmp, f"predictions_{model_name}.csv")
                preds_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path, artifact_path="predictions")

            if not metric_df.empty:
                with tempfile.TemporaryDirectory() as tmp:
                    m_path = os.path.join(tmp, f"metrics_{model_name}.csv")
                    metric_df.to_csv(m_path, index=False)
                    mlflow.log_artifact(m_path, artifact_path="metrics")

        print(f"  MLflow run: {run.info.run_id}  ({len(preds_df):,} predictions logged)")

    return preds_df


# ---------------------------------------------------------------------------
# Write predictions to gold.fact_predictions
# ---------------------------------------------------------------------------

def _write_predictions(all_preds: list[pd.DataFrame], engine) -> None:
    if not all_preds:
        return
    combined = pd.concat([p for p in all_preds if not p.empty], ignore_index=True)
    if combined.empty:
        print("  No predictions to write.")
        return

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

    combined.to_sql(
        "fact_predictions", engine, schema="gold",
        if_exists="append", index=False, chunksize=2_000,
    )
    print(f"  {len(combined):,} rows → gold.fact_predictions")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(
    register: bool = True,
    filter_indicator: str | None = None,
) -> None:
    cfg = _load_config()

    mlflow.set_tracking_uri(cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(cfg.get("experiment_name", "finance-price-forecasting"))

    test_size = int(cfg.get("training", {}).get("test_size", 1))

    engine = _get_engine()

    print("=== Step 4 — ML Training (MLflow) ===")
    print(f"  Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"  Experiment   : {cfg.get('experiment_name')}")

    print("\n  Loading data from gold.fact_prices...")
    df = _load_indicators(engine)

    if df.empty:
        print("  gold.fact_prices is empty — run main.py first, then re-run train.py.")
        engine.dispose()
        return

    if filter_indicator:
        df = df[df["indicator"] == filter_indicator]

    print(f"  {len(df):,} rows | {df['country_id'].nunique()} assets "
          f"| {df['indicator'].nunique()} indicators\n")

    model_types = ["linear_trend", "holt_smoothing"]
    all_preds: list[pd.DataFrame] = []

    for model_name in model_types:
        print(f"--- {model_name} ---")
        preds = _run_experiment(df, model_name, cfg, test_size, register)
        all_preds.append(preds)

    print("\n=== Writing gold.fact_predictions ===")
    _write_predictions(all_preds, engine)

    engine.dispose()

    print(f"\n  Done. View results:")
    print(f"    mlflow ui --port 5001")
    print(f"    open http://localhost:5001")


if __name__ == "__main__":
    load_dotenv(_PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="MLflow-tracked finance forecasting")
    parser.add_argument("--no-register",  action="store_true", help="skip model registry")
    parser.add_argument("--indicator",    default=None,        help="train one indicator only")
    args = parser.parse_args()
    train(register=not args.no_register, filter_indicator=args.indicator)
