"""
spark_eda.py — Exploratory Data Analysis on the long-format Spark DataFrame.

Analyses produced:
    null_profile      — null count + % per column
    indicator_stats   — count, mean, stddev, min, p25, median, p75, max per indicator
    year_coverage     — distinct countries + indicators + total rows per year
    country_coverage  — year span and row count per country
    region_counts     — country count + row count per region

Usage:
    from etl.spark_eda import run_eda

    summaries = run_eda(df)             # prints formatted report, returns dict of DFs
    summaries['indicator_stats'].show() # access individual results
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Individual EDA functions
# ---------------------------------------------------------------------------

def null_profile(df: DataFrame) -> DataFrame:
    """Return a DataFrame with null count and null % for every column."""
    total = df.count()
    agg_exprs = [
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]
    null_row = df.agg(*agg_exprs).collect()[0]
    rows = [
        (c, int(null_row[c]), round(null_row[c] / total * 100, 2))
        for c in df.columns
    ]
    return df.sparkSession.createDataFrame(rows, ["column", "null_count", "null_pct"])


def indicator_stats(df: DataFrame) -> DataFrame:
    """
    Descriptive statistics per indicator.

    Columns: indicator, count, mean, stddev, min, p25, median, p75, max
    """
    return (
        df.groupBy("indicator")
        .agg(
            F.count("value").alias("count"),
            F.round(F.mean("value"),                              4).alias("mean"),
            F.round(F.stddev("value"),                            4).alias("stddev"),
            F.round(F.min("value"),                               4).alias("min"),
            F.round(F.percentile_approx("value", 0.25).cast("double"), 4).alias("p25"),
            F.round(F.percentile_approx("value", 0.50).cast("double"), 4).alias("median"),
            F.round(F.percentile_approx("value", 0.75).cast("double"), 4).alias("p75"),
            F.round(F.max("value"),                               4).alias("max"),
        )
        .orderBy("indicator")
    )


def year_coverage(df: DataFrame) -> DataFrame:
    """Distinct countries, indicators, and total rows available per year."""
    return (
        df.groupBy("year")
        .agg(
            F.countDistinct("iso_code").alias("country_count"),
            F.countDistinct("indicator").alias("indicator_count"),
            F.count("value").alias("row_count"),
        )
        .orderBy("year")
    )


def country_coverage(df: DataFrame) -> DataFrame:
    """
    Year span and data density per country.

    Sorted by year_count descending so the best-covered countries appear first.
    """
    return (
        df.groupBy("iso_code", "country_name", "region")
        .agg(
            F.count("value").alias("row_count"),
            F.min("year").alias("year_from"),
            F.max("year").alias("year_to"),
            F.countDistinct("year").alias("year_count"),
            F.countDistinct("indicator").alias("indicator_count"),
        )
        .orderBy(F.desc("year_count"), F.desc("row_count"))
    )


def region_counts(df: DataFrame) -> DataFrame:
    """Country count and row count per region/continent."""
    return (
        df.groupBy("region")
        .agg(
            F.countDistinct("iso_code").alias("country_count"),
            F.count("value").alias("row_count"),
        )
        .orderBy(F.desc("row_count"))
    )


def source_coverage(df: DataFrame) -> DataFrame:
    """Row count and distinct indicators per data source."""
    return (
        df.groupBy("source")
        .agg(
            F.count("value").alias("row_count"),
            F.countDistinct("indicator").alias("indicator_count"),
            F.countDistinct("iso_code").alias("country_count"),
            F.min("year").alias("year_from"),
            F.max("year").alias("year_to"),
        )
        .orderBy(F.desc("row_count"))
    )


# ---------------------------------------------------------------------------
# Full EDA runner
# ---------------------------------------------------------------------------

def run_eda(df: DataFrame) -> dict[str, DataFrame]:
    """
    Run all EDA checks, print a formatted report, and return named DataFrames.

    The returned DataFrames are not yet collected — call .show() or .toPandas()
    on any of them to inspect further.

    Returns:
        {
            'null_profile':     null count/% per column,
            'indicator_stats':  descriptive stats per indicator,
            'year_coverage':    countries + indicators per year,
            'country_coverage': year span per country,
            'region_counts':    country/row counts per region,
            'source_coverage':  coverage breakdown by source dataset,
        }
    """
    print("\n" + "=" * 60)
    print("EDA — Exploratory Data Analysis")
    print("=" * 60)

    total_rows = df.count()
    n_cols     = len(df.columns)
    n_countries  = df.select("iso_code").distinct().count()
    n_indicators = df.select("indicator").distinct().count()
    year_min, year_max = df.agg(F.min("year"), F.max("year")).collect()[0]

    print(f"\n  Total rows      : {total_rows:,}")
    print(f"  Columns         : {n_cols}")
    print(f"  Countries       : {n_countries}")
    print(f"  Indicators      : {n_indicators}")
    print(f"  Year range      : {year_min} – {year_max}")
    print(f"  Schema          : {', '.join(df.columns)}")

    print("\n--- [1/6] Null Profile ---")
    ndf = null_profile(df)
    ndf.show(truncate=False)

    print("--- [2/6] Indicator Statistics ---")
    istats = indicator_stats(df)
    istats.show(50, truncate=False)

    print("--- [3/6] Year Coverage ---")
    ycov = year_coverage(df)
    ycov.show(80, truncate=False)

    print("--- [4/6] Country Coverage (top 25) ---")
    ccov = country_coverage(df)
    ccov.show(25, truncate=False)

    print("--- [5/6] Rows by Region ---")
    rcounts = region_counts(df)
    rcounts.show(truncate=False)

    print("--- [6/6] Coverage by Source Dataset ---")
    scov = source_coverage(df)
    scov.show(truncate=False)

    return {
        "null_profile":     ndf,
        "indicator_stats":  istats,
        "year_coverage":    ycov,
        "country_coverage": ccov,
        "region_counts":    rcounts,
        "source_coverage":  scov,
    }
