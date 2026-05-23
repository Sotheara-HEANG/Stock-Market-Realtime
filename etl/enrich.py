"""
enrich.py - derived stock price features.
"""

from __future__ import annotations

import warnings

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

warnings.filterwarnings("ignore")

CATEGORY_BY_SYMBOL = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "AMZN": "Consumer Cyclical",
    "TSLA": "Consumer Cyclical",
    "JPM": "Financial Services",
    "BAC": "Financial Services",
    "V": "Financial Services",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "XOM": "Energy",
}


def add_commodity_category(wide_df: DataFrame) -> DataFrame:
    """Add a market sector/category from the tracked symbol."""
    category_map = F.create_map(
        *[x for pair in ((F.lit(k), F.lit(v)) for k, v in CATEGORY_BY_SYMBOL.items()) for x in pair]
    )
    df = wide_df.withColumn(
        "commodity_category",
        F.coalesce(category_map[F.col("country_code")], F.lit("Other")),
    )
    mapped = df.filter(F.col("commodity_category").isNotNull()).count()
    print(f"  stock sector mapped: {mapped:,} rows")
    return df


def add_price_movement(wide_df: DataFrame) -> DataFrame:
    """Add absolute and percentage movement versus previous period close."""
    if "close_price" not in wide_df.columns:
        print("  price movement: close_price column missing, skipped")
        return (
            wide_df
            .withColumn("price_change", F.lit(None).cast("double"))
            .withColumn("price_change_pct", F.lit(None).cast("double"))
            .withColumn("price_trend", F.lit(None).cast("string"))
        )

    # Partition by both country_code and timeframe, ordered by time_index
    w = Window.partitionBy("country_code", "timeframe").orderBy("time_index")
    df = wide_df.withColumn("previous_close", F.lag("close_price").over(w))
    df = df.withColumn(
        "price_change",
        F.when(F.col("previous_close").isNotNull(), F.round(F.col("close_price") - F.col("previous_close"), 4)),
    )
    df = df.withColumn(
        "price_change_pct",
        F.when(
            F.col("previous_close").isNotNull() & (F.col("previous_close") != 0),
            F.round((F.col("price_change") / F.col("previous_close")) * 100, 4),
        ),
    )
    df = df.withColumn(
        "price_trend",
        F.when(F.col("price_change_pct") >= 1, F.lit("up"))
         .when(F.col("price_change_pct") <= -1, F.lit("down"))
         .when(F.col("price_change_pct").isNotNull(), F.lit("flat"))
         .otherwise(F.lit("new")),
    ).drop("previous_close")

    dist = df.groupBy("price_trend").count().orderBy("price_trend").collect()
    dist_str = ", ".join(f"{row['price_trend']}:{row['count']}" for row in dist)
    print(f"  price trend distribution: {dist_str}")
    return df


def add_volatility(wide_df: DataFrame) -> DataFrame:
    """Add intraperiod price range and simple volatility label."""
    needed = {"high_price", "low_price", "close_price"}
    missing = needed - set(wide_df.columns)
    if missing:
        print(f"  volatility: missing columns {missing}, skipped")
        return (
            wide_df
            .withColumn("intraday_range", F.lit(None).cast("double"))
            .withColumn("intraday_range_pct", F.lit(None).cast("double"))
            .withColumn("volatility_level", F.lit(None).cast("string"))
        )

    df = wide_df.withColumn("intraday_range", F.round(F.col("high_price") - F.col("low_price"), 4))
    df = df.withColumn(
        "intraday_range_pct",
        F.when(F.col("close_price") != 0, F.round((F.col("intraday_range") / F.col("close_price")) * 100, 4)),
    )
    df = df.withColumn(
        "volatility_level",
        F.when(F.col("intraday_range_pct") >= 8, F.lit("high"))
         .when(F.col("intraday_range_pct") >= 3, F.lit("medium"))
         .when(F.col("intraday_range_pct").isNotNull(), F.lit("low"))
         .otherwise(F.lit(None).cast("string")),
    )
    print("  volatility metrics added")
    return df


def add_category_averages(wide_df: DataFrame) -> DataFrame:
    """Add category-level average close price and count."""
    needed = {"commodity_category", "timeframe", "time_index", "close_price"}
    missing = needed - set(wide_df.columns)
    if missing:
        print(f"  category averages: missing columns {missing}, skipped")
        return (
            wide_df
            .withColumn("category_avg_close", F.lit(None).cast("double"))
            .withColumn("category_count", F.lit(None).cast("int"))
        )

    agg_df = (
        wide_df
        .groupBy("commodity_category", "timeframe", "time_index")
        .agg(
            F.round(F.avg("close_price"), 4).alias("category_avg_close"),
            F.count("*").alias("category_count"),
        )
    )
    df = wide_df.join(agg_df, on=["commodity_category", "timeframe", "time_index"], how="left")
    print(f"  category averages: {agg_df.count():,} category/timeframe/time_index groups")
    return df


def enrich(wide_df: DataFrame, spark: SparkSession) -> DataFrame:
    """Apply stock price enrichments to the cleaned wide DataFrame."""
    _ = spark
    print("=== Enrich ===")

    print("  [1/4] Stock sector...")
    df = add_commodity_category(wide_df)

    print("  [2/4] Price movement...")
    df = add_price_movement(df)

    print("  [3/4] Volatility...")
    df = add_volatility(df)

    print("  [4/4] Category averages...")
    df = add_category_averages(df)

    new_cols = [
        "commodity_category", "price_change", "price_change_pct", "price_trend",
        "intraday_range", "intraday_range_pct", "volatility_level",
        "category_avg_close", "category_count",
    ]
    print(f"\n  Added columns: {new_cols}")
    print(f"  Final shape: {df.count():,} rows x {len(df.columns)} columns")
    return df
