"""
main.py - run the Finnhub trading pipeline end to end.

    Extract → Transform → Enrich → Load → Predict → MLflow Train

Usage:
    python main.py                # Finnhub stock data (default)
    python main.py --legacy       # accepted for backwards compatibility
    python main.py --api          # accepted for backwards compatibility
    python main.py --no-predict   # skip predict + MLflow steps
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
load_dotenv()

from etl.extract import extract_all
from etl.transform import (
    get_spark,
    to_spark_dict,
    union_sources,
    normalize_country_names,
    pivot_wide,
    drop_missing_price,
)
from etl.enrich import enrich
from etl.load import load
from etl.predict import predict

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent / "04_ml" / "training"))
from train import train as mlflow_train

if __name__ == "__main__":
    include_legacy   = "--legacy" in sys.argv
    include_api      = "--api" in sys.argv
    skip_predict     = "--no-predict" in sys.argv
    spark = get_spark()

    # ------------------------------------------------------------------
    # Step 1 — Extract
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Step 1/5 — Extract Finnhub market data")
    print("=" * 50)
    pdfs = extract_all(
        include_realtime=True,
        include_legacy=include_legacy,
        include_api=include_api,
    )

    # ------------------------------------------------------------------
    # Step 2 — Transform
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Step 2/5 — Transform")
    print("=" * 50)
    spark_dfs = to_spark_dict(pdfs, spark)
    long_df   = union_sources(spark_dfs)
    if include_legacy:
        long_df = normalize_country_names(long_df)
    wide_df   = pivot_wide(long_df)
    wide_df   = drop_missing_price(wide_df)

    # ------------------------------------------------------------------
    # Step 3 — Enrich
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Step 3/5 — Enrich")
    print("=" * 50)
    enriched_df = enrich(wide_df, spark)

    # ------------------------------------------------------------------
    # Step 4 — Load
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Step 4/5 — Load")
    print("=" * 50)
    load(pdfs, enriched_df)

    spark.stop()

    # ------------------------------------------------------------------
    # Step 5 — Predict + MLflow Training
    # ------------------------------------------------------------------
    if not skip_predict:
        print("\n" + "=" * 50)
        print("Step 5a/5 — Predict")
        print("=" * 50)
        predict()

        print("\n" + "=" * 50)
        print("Step 5b/5 — MLflow Training")
        print("=" * 50)
        mlflow_train()
    else:
        print("\nStep 5/5 — Predict + MLflow  [skipped]")
