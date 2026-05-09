"""
spark_aggregate.py — data aggregation on the long-format Spark DataFrame.

Four aggregation layers:
    1. by_region_year      — avg/stddev/min/max per (region, year, indicator)
    2. by_indicator_year   — global trend: avg value per (indicator, year)
    3. top_countries       — top and bottom N countries per indicator (latest year)
    4. yoy_change          — year-over-year absolute and % change per (country, indicator)

Usage:
    from etl.spark_aggregate import aggregate

    results = aggregate(df)
    results['by_region_year'].filter(...).show()

    # Optionally save all to parquet:
    results = aggregate(df, out_dir='data/aggregations')
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# 1. Regional aggregation
# ---------------------------------------------------------------------------

def by_region_year(df: DataFrame) -> DataFrame:
    """
    Average, stddev, min, and max value per (region, year, indicator).

    Rows with a null region are excluded so aggregate counts are not skewed
    by unmapped Polity5 scodes or historical entities.
    """
    return (
        df.filter(F.col("region").isNotNull())
        .groupBy("region", "year", "indicator", "unit")
        .agg(
            F.count("value").alias("country_count"),
            F.round(F.avg("value"),    4).alias("avg_value"),
            F.round(F.stddev("value"), 4).alias("stddev_value"),
            F.round(F.min("value"),    4).alias("min_value"),
            F.round(F.max("value"),    4).alias("max_value"),
        )
        .orderBy("region", "indicator", "year")
    )


# ---------------------------------------------------------------------------
# 2. Global trend by indicator
# ---------------------------------------------------------------------------

def by_indicator_year(df: DataFrame) -> DataFrame:
    """
    Global trend: average value per (indicator, year) across all countries.

    Useful for plotting time-series of each indicator at world level.
    """
    return (
        df.groupBy("indicator", "year", "unit")
        .agg(
            F.count("value").alias("country_count"),
            F.round(F.avg("value"),    4).alias("avg_value"),
            F.round(F.stddev("value"), 4).alias("stddev_value"),
            F.round(F.min("value"),    4).alias("min_value"),
            F.round(F.max("value"),    4).alias("max_value"),
        )
        .orderBy("indicator", "year")
    )


# ---------------------------------------------------------------------------
# 3. Top / bottom countries (latest year)
# ---------------------------------------------------------------------------

def top_countries(df: DataFrame, n: int = 10) -> DataFrame:
    """
    Top and bottom N countries per indicator for the most recent year in the data.

    Output columns:
        rank_type   — 'top' or 'bottom'
        rank        — 1 = highest (for top) or lowest (for bottom)
        indicator, country_name, iso_code, region, value, year
    """
    latest_year = df.agg(F.max("year")).collect()[0][0]
    latest = df.filter(F.col("year") == latest_year)

    w_desc = Window.partitionBy("indicator").orderBy(F.desc("value"))
    w_asc  = Window.partitionBy("indicator").orderBy(F.asc("value"))

    top = (
        latest
        .withColumn("rank", F.row_number().over(w_desc))
        .filter(F.col("rank") <= n)
        .withColumn("rank_type", F.lit("top"))
    )
    bottom = (
        latest
        .withColumn("rank", F.row_number().over(w_asc))
        .filter(F.col("rank") <= n)
        .withColumn("rank_type", F.lit("bottom"))
    )

    return (
        top.unionAll(bottom)
        .select(
            "rank_type", "rank", "indicator", "unit",
            "country_name", "iso_code", "region", "value", "year",
        )
        .orderBy("indicator", "rank_type", "rank")
    )


# ---------------------------------------------------------------------------
# 4. Year-over-year change
# ---------------------------------------------------------------------------

def yoy_change(df: DataFrame) -> DataFrame:
    """
    Year-over-year absolute and % change per (country, indicator).

    Output columns:
        iso_code, country_name, region, indicator, year,
        value, prev_value, yoy_abs, yoy_pct

    Rows where prev_value is null (first year of series) or zero are excluded
    to avoid division errors.
    """
    w = Window.partitionBy("iso_code", "indicator").orderBy("year")
    prev_val = F.lag("value").over(w)

    return (
        df.withColumn("prev_value", prev_val)
        .filter(F.col("prev_value").isNotNull() & (F.col("prev_value") != 0))
        .withColumn("yoy_abs", F.round(F.col("value") - F.col("prev_value"), 4))
        .withColumn(
            "yoy_pct",
            F.round(
                (F.col("value") - F.col("prev_value")) / F.abs(F.col("prev_value")) * 100,
                4,
            ),
        )
        .select(
            "iso_code", "country_name", "region",
            "indicator", "unit", "year",
            "value", "prev_value", "yoy_abs", "yoy_pct",
        )
        .orderBy("indicator", F.desc("yoy_pct"))
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def aggregate(
    df: DataFrame,
    out_dir: str | None = None,
    sample_indicator: str = "gdp_growth_pct",
    sample_year_from: int = 2015,
) -> dict[str, DataFrame]:
    """
    Run all four aggregations, print sample outputs, and optionally save to parquet.

    Args:
        df               : long-format Spark DataFrame from extract_from_db()
        out_dir          : if given, write each result as parquet under this path
        sample_indicator : indicator used in the printed sample outputs
        sample_year_from : earliest year shown in the regional sample

    Returns:
        dict of named Spark DataFrames (lazy — trigger actions with .show() / .toPandas())
    """
    print("\n" + "=" * 60)
    print("Aggregations")
    print("=" * 60)

    print("\n  [1/4] By region + year (per indicator)...")
    reg_year = by_region_year(df)

    print("  [2/4] Global trend by indicator + year...")
    ind_year = by_indicator_year(df)

    print("  [3/4] Top / bottom 10 countries per indicator (latest year)...")
    top = top_countries(df, n=10)

    print("  [4/4] Year-over-year change...")
    yoy = yoy_change(df)

    results: dict[str, DataFrame] = {
        "by_region_year":    reg_year,
        "by_indicator_year": ind_year,
        "top_countries":     top,
        "yoy_change":        yoy,
    }

    # --- Optional parquet export ---
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        for name, agg_df in results.items():
            dest = str(p / name)
            agg_df.write.mode("overwrite").parquet(dest)
            print(f"  Saved {name} → {dest}/")

    # --- Sample prints ---
    print(f"\n--- Regional averages: {sample_indicator} ({sample_year_from}+) ---")
    reg_year.filter(
        (F.col("indicator") == sample_indicator) & (F.col("year") >= sample_year_from)
    ).show(40, truncate=False)

    print(f"\n--- Global trend: {sample_indicator} ---")
    ind_year.filter(F.col("indicator") == sample_indicator).show(30, truncate=False)

    print(f"\n--- Top 5 countries by {sample_indicator} (latest year) ---")
    top.filter(
        (F.col("indicator") == sample_indicator) & (F.col("rank_type") == "top") & (F.col("rank") <= 5)
    ).show(truncate=False)

    print(f"\n--- Biggest YoY gains: {sample_indicator} ---")
    yoy.filter(F.col("indicator") == sample_indicator).show(10, truncate=False)

    return results
