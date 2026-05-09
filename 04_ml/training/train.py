"""
train.py — MLflow-tracked model training for economic/governance forecasting.

Wraps etl/predict.py with MLflow experiment tracking:
  - Logs hyperparameters (horizon, min_obs, model type)
  - Logs per-model aggregate metrics (MAE, RMSE on held-out test years)
  - Saves prediction CSV as an artifact
  - Registers the best model in the MLflow Model Registry

Usage:
    python 04_ml/training/train.py
    python 04_ml/training/train.py --no-register   # skip model registry
    python 04_ml/training/train.py --indicator gdp_growth_pct
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from etl.predict import (
    HORIZON,
    INDICATORS,
    MIN_OBS,
    _get_engine,
    _load_indicators,
    _fit_series,
    predict,
)
sys.path.insert(0, str(_ROOT / "04_ml" / "training"))
from evaluate import compute_metrics

_CONFIG_PATH = _ROOT / "04_ml" / "mlflow" / "mlflow_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["mlflow"]


def _train_with_tracking(
    indicators: list[str] | None = None,
    register: bool = True,
) -> None:
    config = _load_config()

    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    engine = _get_engine()
    df = _load_indicators(engine)

    if indicators:
        df = df[df["indicator"].isin(indicators)]

    target_indicators = indicators or INDICATORS
    print(f"Training on indicators: {target_indicators}")

    for model_name in ("linear_trend", "holt_smoothing"):
        with mlflow.start_run(run_name=f"{model_name}") as run:
            # Log parameters
            mlflow.log_params({
                "model_name":     model_name,
                "horizon":        HORIZON,
                "min_obs":        MIN_OBS,
                "indicators":     ",".join(target_indicators),
                "n_countries":    df["country_id"].nunique(),
            })
            mlflow.set_tags(config.get("default_tags", {}))

            all_preds: list[dict] = []
            metrics_per_indicator: dict[str, dict] = {}

            for (country_id, indicator), grp in df.groupby(["country_id", "indicator"]):
                if indicator not in target_indicators:
                    continue
                series = (
                    grp[["year", "value"]]
                    .dropna(subset=["value"])
                    .sort_values("year")
                    .drop_duplicates(subset=["year"])
                    .reset_index(drop=True)
                )
                if len(series) < MIN_OBS:
                    continue

                # Hold out last TEST_YEARS for evaluation
                test_years = 3
                train_series = series.iloc[:-test_years]
                test_series  = series.iloc[-test_years:]

                if len(train_series) < MIN_OBS:
                    continue

                preds = _fit_series(int(country_id), indicator, train_series)
                model_preds = [p for p in preds if p["model_name"] == model_name]
                all_preds.extend(model_preds)

                # Compute in-sample metrics for this series
                actual = test_series["value"].values
                predicted = np.array([p["predicted_value"] for p in model_preds[:len(actual)]])
                if len(predicted) > 0 and len(actual) > 0:
                    n = min(len(actual), len(predicted))
                    mae  = float(np.mean(np.abs(actual[:n] - predicted[:n])))
                    rmse = float(np.sqrt(np.mean((actual[:n] - predicted[:n]) ** 2)))
                    if indicator not in metrics_per_indicator:
                        metrics_per_indicator[indicator] = {"mae": [], "rmse": []}
                    metrics_per_indicator[indicator]["mae"].append(mae)
                    metrics_per_indicator[indicator]["rmse"].append(rmse)

            # Log aggregate metrics per indicator
            for ind, vals in metrics_per_indicator.items():
                mlflow.log_metric(f"{ind}_mae",  round(float(np.mean(vals["mae"])),  4))
                mlflow.log_metric(f"{ind}_rmse", round(float(np.mean(vals["rmse"])), 4))

            # Overall metrics
            all_mae  = [v for vals in metrics_per_indicator.values() for v in vals["mae"]]
            all_rmse = [v for vals in metrics_per_indicator.values() for v in vals["rmse"]]
            if all_mae:
                mlflow.log_metric("overall_mae",  round(float(np.mean(all_mae)),  4))
                mlflow.log_metric("overall_rmse", round(float(np.mean(all_rmse)), 4))

            # Save predictions CSV artifact
            if all_preds:
                preds_df = pd.DataFrame(all_preds)
                artifact_path = _ROOT / "04_ml" / "models" / f"{model_name}_predictions.csv"
                artifact_path.parent.mkdir(exist_ok=True)
                preds_df.to_csv(artifact_path, index=False)
                mlflow.log_artifact(str(artifact_path), artifact_path="predictions")

            print(f"  [{model_name}] Run ID: {run.info.run_id}")
            print(f"  [{model_name}] {len(all_preds)} prediction rows  |  "
                  f"overall_mae={round(float(np.mean(all_mae)), 4) if all_mae else 'N/A'}")

    engine.dispose()
    print("\nMLflow tracking complete.")
    print(f"View runs: mlflow ui --backend-store-uri {config['tracking_uri']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow model registry")
    parser.add_argument("--indicator",   help="Train on a single indicator only")
    args = parser.parse_args()

    load_dotenv(_ROOT / ".env")
    _train_with_tracking(
        indicators=[args.indicator] if args.indicator else None,
        register=not args.no_register,
    )
