"""
batch_job.py — PySpark batch job: Bronze → Silver → Gold.

Reads from PostgreSQL bronze.raw_indicators, applies the full
Transform → Enrich pipeline, and writes cleaned data to the
silver and gold schemas.

Usage:
    spark-submit \
        --packages org.postgresql:postgresql:42.6.0 \
        02_etl/batch/pyspark_batch/batch_job.py

    # Dry run — print row counts only, no writes
    python 02_etl/batch/pyspark_batch/batch_job.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from etl.transform import (
    get_spark,
    normalize_country_names,
    pivot_wide,
    drop_missing_gdp_hdi,
)
from etl.enrich import enrich

DB_HOST     = os.environ.get("DB_HOST", "localhost")
DB_PORT     = os.environ.get("DB_PORT", "5432")
DB_NAME     = os.environ.get("DB_NAME", "econ_pipeline")
DB_USER     = os.environ.get("DB_USER", "kongsattha")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_URL      = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"

_JDBC_OPTS = {
    "url":      DB_URL,
    "user":     DB_USER,
    "password": DB_PASSWORD,
    "driver":   "org.postgresql.Driver",
}


def _read_bronze(spark: SparkSession) -> DataFrame:
    """Load raw_indicators from the bronze schema into Spark."""
    return (
        spark.read.format("jdbc")
        .options(**_JDBC_OPTS, dbtable="bronze.raw_indicators")
        .load()
    )


def _write_silver(df: DataFrame, dry_run: bool) -> None:
    """Write the long-format normalised DataFrame to silver.indicators."""
    print(f"  Silver: {df.count():,} rows")
    if dry_run:
        df.show(5, truncate=False)
        return
    (
        df.write.format("jdbc")
        .options(**_JDBC_OPTS, dbtable="silver.indicators")
        .mode("overwrite")
        .save()
    )


def _write_gold(df: DataFrame, dry_run: bool) -> None:
    """Write the wide enriched DataFrame to gold.fact_indicators."""
    print(f"  Gold:   {df.count():,} rows")
    if dry_run:
        df.show(5, truncate=True)
        return
    (
        df.write.format("jdbc")
        .options(**_JDBC_OPTS, dbtable="gold.fact_indicators")
        .mode("overwrite")
        .save()
    )


def run(dry_run: bool = False) -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 50)
    print("Batch Job — Bronze → Silver → Gold")
    print("=" * 50)

    # ── Bronze → Silver ──────────────────────────────────────────────────
    print("\nStep 1/3  Reading Bronze layer...")
    bronze_df = _read_bronze(spark)
    print(f"  Bronze: {bronze_df.count():,} rows")

    print("\nStep 2/3  Transforming to Silver (normalise → pivot → filter)...")
    silver_long = normalize_country_names(bronze_df)

    # Pivot to wide format (one row per country × year)
    wide_df = pivot_wide(silver_long)
    wide_df = drop_missing_gdp_hdi(wide_df)
    _write_silver(silver_long, dry_run)

    # ── Silver → Gold ─────────────────────────────────────────────────────
    print("\nStep 3/3  Enriching to Gold (derived features + regional averages)...")
    gold_df = enrich(wide_df, spark)
    _write_gold(gold_df, dry_run)

    spark.stop()
    print("\nBatch job complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print counts only, no DB writes")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
