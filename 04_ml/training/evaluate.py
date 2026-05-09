"""
evaluate.py — model evaluation utilities for the forecasting pipeline.

Computes regression metrics comparing model predictions against held-out actuals.

Usage:
    from 04_ml.training.evaluate import compute_metrics, print_report
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and R² between actual and predicted arrays.

    Args:
        actual:    Array of ground-truth values.
        predicted: Array of model predictions (same length as actual).

    Returns:
        Dict with keys: mae, rmse, mape, r2
    """
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    n = min(len(actual), len(predicted))
    if n == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    a = actual[:n]
    p = predicted[:n]

    mae  = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))

    # MAPE — skip zero actuals to avoid division by zero
    nonzero = a != 0
    mape = float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100) if nonzero.any() else np.nan

    # R²
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4), "r2": round(r2, 4)}


def evaluate_all(preds_df: pd.DataFrame, actuals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate all (model_name, indicator) combinations.

    Args:
        preds_df:   DataFrame with columns: country_id, indicator, model_name,
                    predicted_year, predicted_value
        actuals_df: DataFrame with columns: country_id, indicator, year, value

    Returns:
        DataFrame with columns: model_name, indicator, mae, rmse, mape, r2, n_series
    """
    records = []

    for (model_name, indicator), group in preds_df.groupby(["model_name", "indicator"]):
        actual_ind = actuals_df[actuals_df["indicator"] == indicator]
        merged = group.merge(
            actual_ind.rename(columns={"year": "predicted_year", "value": "actual_value"}),
            on=["country_id", "predicted_year"],
            how="inner",
        )
        if merged.empty:
            continue

        metrics = compute_metrics(
            merged["actual_value"].values,
            merged["predicted_value"].values,
        )
        metrics["model_name"] = model_name
        metrics["indicator"]  = indicator
        metrics["n_series"]   = group["country_id"].nunique()
        records.append(metrics)

    if not records:
        return pd.DataFrame()

    cols = ["model_name", "indicator", "mae", "rmse", "mape", "r2", "n_series"]
    return pd.DataFrame(records)[cols].sort_values(["model_name", "indicator"])


def print_report(eval_df: pd.DataFrame) -> None:
    """Pretty-print the evaluation report."""
    if eval_df.empty:
        print("No evaluation results.")
        return
    print("\n=== Model Evaluation Report ===")
    print(eval_df.to_string(index=False))
    print()
    for model_name, grp in eval_df.groupby("model_name"):
        print(f"{model_name}  →  avg MAE={grp['mae'].mean():.4f}  "
              f"avg RMSE={grp['rmse'].mean():.4f}  avg R²={grp['r2'].mean():.4f}")
