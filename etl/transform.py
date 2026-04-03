"""
transform.py — convert pandas DataFrames to PySpark, merge all sources on country + year.

Pipeline:
    pandas DFs (extract.py)
        → Spark DFs (pandas_to_spark)
        → unified long Spark DF (union_sources)       # all rows, all indicators
        → wide Spark DF (pivot_wide)                  # one row per country+year, indicators as columns

Key notes on join keys:
    - WGI, IMF, V-Dem  : country_code is ISO-3       → join on (country_code, year)
    - Polity5           : country_code is Polity scode → join on (country_code, year) as proxy
    - HDI               : country_code is empty        → only country_name + year available

The wide pivot handles this naturally: rows with the same (country_code, country_name, year)
are merged; HDI rows (empty country_code) appear as separate rows keyed by country_name alone.

Usage:
    from etl.transform import transform

    long_df, wide_df = transform()          # uses offline sources only
    long_df, wide_df = transform(api=True)  # also fetches live World Bank data
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

_spark: Optional[SparkSession] = None


def get_spark() -> SparkSession:
    """Return a singleton SparkSession, creating it on first call."""
    global _spark
    if _spark is None:
        _spark = (
            SparkSession.builder
            .appName("econ-governance-pipeline")
            .config("spark.sql.session.timeZone", "UTC")
            # Reduce default shuffle partitions for local single-node runs
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel("WARN")
    return _spark


# ---------------------------------------------------------------------------
# Schema — all sources share this 6-column structure after extract.py
# ---------------------------------------------------------------------------

LONG_SCHEMA = StructType([
    StructField("country_code", StringType(),  nullable=True),
    StructField("country_name", StringType(),  nullable=True),
    StructField("indicator",    StringType(),  nullable=False),
    StructField("year",         IntegerType(), nullable=False),
    StructField("value",        DoubleType(),  nullable=True),
    StructField("source",       StringType(),  nullable=True),
])


# ---------------------------------------------------------------------------
# Conversion: pandas → Spark
# ---------------------------------------------------------------------------

def pandas_to_spark(pdf: pd.DataFrame, spark: SparkSession) -> DataFrame:
    """
    Convert a single extract.py pandas DataFrame to a Spark DataFrame.

    Coerces dtypes to match LONG_SCHEMA before conversion so Spark doesn't
    infer mixed types (common with object columns containing empty strings).
    """
    pdf = pdf.copy()
    pdf["country_code"] = pdf["country_code"].astype(str).replace("nan", "")
    pdf["country_name"] = pdf["country_name"].astype(str).replace("nan", "")
    pdf["indicator"]    = pdf["indicator"].astype(str)
    pdf["year"]         = pd.to_numeric(pdf["year"], errors="coerce").astype("Int64")
    pdf["value"]        = pd.to_numeric(pdf["value"], errors="coerce")
    pdf["source"]       = pdf["source"].astype(str)

    return spark.createDataFrame(pdf, schema=LONG_SCHEMA)


def to_spark_dict(
    pdfs: dict[str, pd.DataFrame],
    spark: SparkSession,
) -> dict[str, DataFrame]:
    """Convert a dict of pandas DataFrames to a dict of Spark DataFrames."""
    result = {}
    for name, pdf in pdfs.items():
        sdf = pandas_to_spark(pdf, spark)
        print(f"  {name:10s} → Spark  ({sdf.count():,} rows)")
        result[name] = sdf
    return result


# ---------------------------------------------------------------------------
# Merge step 1: union all sources into one long Spark DF
# ---------------------------------------------------------------------------

def union_sources(spark_dfs: dict[str, DataFrame]) -> DataFrame:
    """
    Stack all per-source Spark DataFrames into a single long-format DF.

    All sources share the same 6-column schema, so this is a simple unionAll.
    """
    frames = list(spark_dfs.values())
    if not frames:
        raise ValueError("No DataFrames to union.")

    long_df = frames[0]
    for df in frames[1:]:
        long_df = long_df.unionAll(df)

    total = long_df.count()
    print(f"  Long DF: {total:,} rows total")
    return long_df


# ---------------------------------------------------------------------------
# Merge step 2: pivot long → wide (one row per country + year)
# ---------------------------------------------------------------------------

def pivot_wide(long_df: DataFrame) -> DataFrame:
    """
    Pivot the unified long DF to a wide format:
        (country_code, country_name, year)  →  one column per indicator

    country_name is resolved by taking the most common non-empty name for
    each country_code (coalesces names across sources that share an ISO code).

    HDI rows (empty country_code) are joined on country_name + year instead.
    """
    # --- Resolve a single country_name per country_code ---
    # Some sources have names, others don't; pick the most frequent non-empty one.
    name_resolution = (
        long_df
        .filter(
            (F.col("country_code") != "") &
            (F.col("country_name") != "")
        )
        .groupBy("country_code", "country_name")
        .count()
        .withColumn(
            "rn",
            F.row_number().over(
                __window_by("country_code", order_by="count", desc=True)
            ),
        )
        .filter(F.col("rn") == 1)
        .drop("count", "rn")
    )

    # --- Split: rows that have an ISO-3 country_code vs HDI (empty code) ---
    has_code = long_df.filter(F.col("country_code") != "")
    no_code  = long_df.filter(F.col("country_code") == "")   # HDI only

    # Pivot the ISO-coded rows
    wide_coded = (
        has_code
        .groupBy("country_code", "year")
        .pivot("indicator")
        .agg(F.first("value"))
        .join(name_resolution, on="country_code", how="left")
    )

    # Pivot the name-only rows (HDI)
    wide_hdi = (
        no_code
        .groupBy("country_name", "year")
        .pivot("indicator")
        .agg(F.first("value"))
        .withColumn("country_code", F.lit(""))
    )

    # Align columns so both halves can be unioned
    coded_cols = set(wide_coded.columns)
    hdi_cols   = set(wide_hdi.columns)

    for col in coded_cols - hdi_cols:
        wide_hdi = wide_hdi.withColumn(col, F.lit(None).cast(DoubleType()))
    for col in hdi_cols - coded_cols:
        wide_coded = wide_coded.withColumn(col, F.lit(None).cast(DoubleType()))

    # Use same column order for the union
    all_cols = sorted(
        set(wide_coded.columns) | set(wide_hdi.columns),
        key=lambda c: (c not in ("country_code", "country_name", "year"), c),
    )
    wide_df = wide_coded.select(all_cols).unionAll(wide_hdi.select(all_cols))

    print(f"  Wide DF: {wide_df.count():,} rows  ×  {len(wide_df.columns)} columns")
    return wide_df


def __window_by(partition_col: str, order_by: str, desc: bool = True):
    from pyspark.sql.window import Window
    w = Window.partitionBy(partition_col)
    col = F.col(order_by)
    return w.orderBy(col.desc() if desc else col)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def transform(include_api: bool = False) -> tuple[DataFrame, DataFrame]:
    """
    Run the full transform pipeline.

    1. Extract all sources via extract.py (pandas)
    2. Convert each to a Spark DataFrame
    3. Union into one long Spark DF
    4. Pivot to wide Spark DF (one row per country + year)

    Args:
        include_api: pass True to also fetch live World Bank data.

    Returns:
        (long_df, wide_df) — both are Spark DataFrames.
    """
    from etl.extract import extract_all  # imported here to keep Spark startup lazy

    spark = get_spark()

    print("=== Extract ===")
    pdfs = extract_all(include_api=include_api)

    print("\n=== Pandas → Spark ===")
    spark_dfs = to_spark_dict(pdfs, spark)

    print("\n=== Union sources (long format) ===")
    long_df = union_sources(spark_dfs)

    print("\n=== Pivot wide (merge on country + year) ===")
    wide_df = pivot_wide(long_df)

    print("\n=== Done ===")
    return long_df, wide_df
