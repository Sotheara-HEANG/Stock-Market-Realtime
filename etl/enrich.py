"""
enrich.py — derived features added to the cleaned wide Spark DataFrame.

Three enrichments for real-time financial data, applied in order by enrich():

  1. add_intraday_range()   — (day_high - day_low) / previous_close * 100
  2. add_price_momentum()   — labelled category based on price_change_pct quartile
  3. add_sector_averages()  — per-(sector, year) avg price and avg change_pct

Usage:
    from etl.enrich import enrich
    enriched_df = enrich(wide_df, spark)
"""

from __future__ import annotations

import warnings

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sector map: company name → GICS sector (covers the default symbol list)
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, str] = {
    "Apple Inc":                   "Technology",
    "Microsoft Corp":              "Technology",
    "Alphabet Inc Class A":        "Technology",
    "Amazon.com Inc":              "Consumer Discretionary",
    "NVIDIA Corp":                 "Technology",
    "Meta Platforms Inc":          "Technology",
    "Tesla Inc":                   "Consumer Discretionary",
    "JPMorgan Chase & Co":         "Financials",
    "Johnson & Johnson":           "Healthcare",
    "Visa Inc":                    "Financials",
    "Procter & Gamble Co":         "Consumer Staples",
    "UnitedHealth Group Inc":      "Healthcare",
    "Home Depot Inc":              "Consumer Discretionary",
    "Mastercard Inc":              "Financials",
    "Walt Disney Co":              "Communication Services",
    "Netflix Inc":                 "Communication Services",
    "Adobe Inc":                   "Technology",
    "Salesforce Inc":              "Technology",
    "PayPal Holdings Inc":         "Financials",
    "Bank of America Corp":        "Financials",
}


# ---------------------------------------------------------------------------
# 1. Intraday range
# ---------------------------------------------------------------------------

def add_intraday_range(wide_df: DataFrame) -> DataFrame:
    """
    intraday_range_pct = (day_high - day_low) / previous_close * 100

    Measures daily volatility as a percentage of the prior close.
    """
    needed = {"day_high_usd", "day_low_usd", "previous_close_usd"}
    if not needed.issubset(wide_df.columns):
        print(f"  intraday_range_pct: missing columns {needed - set(wide_df.columns)}, skipped")
        return wide_df.withColumn("intraday_range_pct", F.lit(None).cast("double"))

    df = wide_df.withColumn(
        "intraday_range_pct",
        F.when(
            F.col("previous_close_usd").isNotNull() & (F.col("previous_close_usd") != 0),
            F.round(
                (F.col("day_high_usd") - F.col("day_low_usd"))
                / F.col("previous_close_usd") * 100,
                4,
            ),
        ).otherwise(F.lit(None).cast("double")),
    )

    non_null = df.filter(F.col("intraday_range_pct").isNotNull()).count()
    print(f"  intraday_range_pct: {non_null:,} non-null values")
    return df


# ---------------------------------------------------------------------------
# 2. Price momentum category
# ---------------------------------------------------------------------------

def add_price_momentum(wide_df: DataFrame) -> DataFrame:
    """
    price_momentum = categorical label derived from price_change_pct:
        strong_up    ≥ +2 %
        up           ≥ +0.5 %
        flat         > −0.5 %
        down         > −2 %
        strong_down  ≤ −2 %
    """
    if "price_change_pct" not in wide_df.columns:
        print("  price_momentum: price_change_pct column missing, skipped")
        return wide_df.withColumn("price_momentum", F.lit(None).cast("string"))

    df = wide_df.withColumn(
        "price_momentum",
        F.when(F.col("price_change_pct") >= 2.0,   F.lit("strong_up"))
         .when(F.col("price_change_pct") >= 0.5,   F.lit("up"))
         .when(F.col("price_change_pct") > -0.5,   F.lit("flat"))
         .when(F.col("price_change_pct") > -2.0,   F.lit("down"))
         .when(F.col("price_change_pct").isNotNull(), F.lit("strong_down"))
         .otherwise(F.lit(None).cast("string")),
    )

    dist = (
        df.filter(F.col("price_momentum").isNotNull())
        .groupBy("price_momentum")
        .count()
        .orderBy("count", ascending=False)
        .collect()
    )
    dist_str = ", ".join(f"{r['price_momentum']}:{r['count']}" for r in dist)
    print(f"  price_momentum distribution: {dist_str}")
    return df


# ---------------------------------------------------------------------------
# 3. Sector averages
# ---------------------------------------------------------------------------

def add_sector_averages(wide_df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Join a sector lookup and add per-(sector, year) averages for
    current_price_usd and price_change_pct.

    Output columns:
        sector                  — GICS sector string
        sector_avg_price        — avg current_price_usd by (sector, year)
        sector_avg_change_pct   — avg price_change_pct  by (sector, year)
    """
    sector_rows = list(_SECTOR_MAP.items())
    sector_df = spark.createDataFrame(sector_rows, ["country_name", "sector"])

    df = wide_df.join(sector_df, on="country_name", how="left")

    for col_name, alias in [
        ("current_price_usd", "sector_avg_price"),
        ("price_change_pct",  "sector_avg_change_pct"),
    ]:
        if col_name not in df.columns:
            continue
        agg_df = (
            df.filter(F.col(col_name).isNotNull() & F.col("sector").isNotNull())
            .groupBy("sector", "year")
            .agg(F.round(F.avg(col_name), 4).alias(alias))
        )
        df = df.join(agg_df, on=["sector", "year"], how="left")

    covered   = df.filter(F.col("sector").isNotNull()).count()
    uncovered = df.count() - covered
    print(f"  sector mapped: {covered:,} rows  ({uncovered:,} unmapped)")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def enrich(wide_df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Apply all three financial enrichments to the cleaned wide DataFrame.

    Args:
        wide_df : cleaned wide Spark DF from transform.transform()
        spark   : active SparkSession

    Returns:
        Enriched Spark DataFrame with additional columns:
            intraday_range_pct, price_momentum,
            sector, sector_avg_price, sector_avg_change_pct
    """
    print("=== Enrich ===")

    print("  [1/3] Intraday range (volatility)...")
    df = add_intraday_range(wide_df)

    print("  [2/3] Price momentum category...")
    df = add_price_momentum(df)

    print("  [3/3] Sector averages...")
    df = add_sector_averages(df, spark)

    new_cols = [
        "intraday_range_pct", "price_momentum",
        "sector", "sector_avg_price", "sector_avg_change_pct",
    ]
    print(f"\n  Added columns: {new_cols}")
    print(f"  Final shape: {df.count():,} rows  ×  {len(df.columns)} columns")
    return df
