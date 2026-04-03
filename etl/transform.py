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
# Clean step 1: normalise country names
# ---------------------------------------------------------------------------

# Canonical names keyed by ISO-3 / Polity scode.
# Derived from inspecting all 31 real conflicts found across WGI, IMF,
# Polity5, V-Dem sources (run: python3 -c "from etl.transform import ...").
CANONICAL_NAMES: dict[str, str] = {
    # Polity5 maps AUS to both Australia and Austria — AUT is the correct code for Austria
    "AUS": "Australia",
    "AUT": "Austria",
    # Congo disambiguation
    "COD": "Democratic Republic of Congo",
    "COG": "Republic of Congo",
    "CON": "Republic of Congo",          # Polity5 scode for COG
    # Cape Verde — official UN name since 2013
    "CPV": "Cabo Verde",
    # Czech Republic → Czechia (official since 2016); CZE also appears as Czechoslovakia in old Polity5 rows
    "CZE": "Czechia",
    # Egypt
    "EGY": "Egypt",
    # Gambia
    "GMB": "Gambia",
    # Germany — GMY is the Polity5 scode; Prussia is historical
    "GMY": "Germany",
    # Hong Kong
    "HKG": "Hong Kong SAR, China",
    # Iran
    "IRN": "Iran",
    # Côte d'Ivoire — IVO is Polity5 scode
    "IVO": "Côte d'Ivoire",
    "CIV": "Côte d'Ivoire",
    # Kyrgyzstan
    "KGZ": "Kyrgyzstan",
    # Korea
    "KOR": "South Korea",
    "PRK": "North Korea",
    # Laos
    "LAO": "Laos",
    # Macao vs Macedonia — MAC is the ISO-3 for Macao; Macedonia is MKD
    "MAC": "Macao SAR, China",
    "MKD": "North Macedonia",
    # Palestine
    "PSE": "Palestine",
    # Russia
    "RUS": "Russia",
    # Sudan — SDN/Sudan-North split handled by keeping Sudan
    "SDN": "Sudan",
    # El Salvador vs Slovenia — SLV is ISO-3 for El Salvador; Slovenia is SVN
    "SLV": "El Salvador",
    "SVN": "Slovenia",
    # Somalia
    "SOM": "Somalia",
    # Slovakia
    "SVK": "Slovakia",
    # Eswatini vs Switzerland — SWZ is ISO-3 for Eswatini; Switzerland is CHE
    "SWZ": "Eswatini",
    "CHE": "Switzerland",
    # Syria
    "SYR": "Syria",
    # Timor-Leste
    "TLS": "Timor-Leste",
    # Turkey / Türkiye — official UN name change 2022; using Turkiye for consistency
    "TUR": "Turkiye",
    # Venezuela
    "VEN": "Venezuela",
    # Vietnam
    "VNM": "Vietnam",
    # Yemen
    "YEM": "Yemen",
    # Yugoslavia — historical; YGS is Polity5 scode
    "YGS": "Yugoslavia",
    # Bahamas
    "BHS": "Bahamas",
    # Bolivia
    "BOL": "Bolivia",
    # Brunei
    "BRN": "Brunei",
    # Kyrgyzstan (alternate Polity5 scode)
    "KYR": "Kyrgyzstan",
    # Serbia and Montenegro (historical)
    "SRB": "Serbia",
}

def normalize_country_names(long_df: DataFrame) -> DataFrame:
    """
    Apply CANONICAL_NAMES to the long DF before pivoting.

    For each row, if the country_code exists in the map, replace country_name
    with the canonical value. Rows whose code is not in the map are unchanged.
    Logs the number of rows updated.
    """
    # Build the map expression here (requires an active SparkSession)
    name_map = F.create_map(
        *[x for pair in ((F.lit(k), F.lit(v)) for k, v in CANONICAL_NAMES.items()) for x in pair]
    )

    corrected = long_df.withColumn(
        "country_name",
        F.coalesce(name_map[F.col("country_code")], F.col("country_name")),
    )

    # Count how many rows had their name changed
    changed = (
        long_df.select("country_code", "country_name")
        .join(
            corrected.select(
                F.col("country_code").alias("cc"),
                F.col("country_name").alias("new_name"),
            ),
            long_df["country_code"] == corrected["country_code"],
            "inner",
        )
        .filter(F.col("country_name") != F.col("new_name"))
        .count()
    )
    # Simpler count: rows whose code is in the canonical map
    mapped_count = long_df.filter(
        F.col("country_code").isin(list(CANONICAL_NAMES.keys()))
    ).count()
    print(f"  Country name normalisation: {mapped_count:,} rows updated to canonical names")
    return corrected


# ---------------------------------------------------------------------------
# Clean step 2: drop rows missing both GDP and HDI
# ---------------------------------------------------------------------------

def drop_missing_gdp_hdi(wide_df: DataFrame) -> DataFrame:
    """
    Drop rows from the wide DF where BOTH gdp_growth_pct AND hdi_value are null.

    These rows have no economic or human-development anchor — they carry only
    political/governance scores and cannot support downstream modelling that
    requires at least one of these two measures.
    """
    before = wide_df.count()
    cleaned = wide_df.filter(
        F.col("gdp_growth_pct").isNotNull() | F.col("hdi_value").isNotNull()
    )
    after = cleaned.count()
    print(f"  Dropped {before - after:,} rows missing both GDP and HDI  ({after:,} remain)")
    return cleaned


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def transform(include_api: bool = False) -> tuple[DataFrame, DataFrame]:
    """
    Run the full transform + clean pipeline.

    1. Extract all sources via extract.py (pandas)
    2. Convert each to a Spark DataFrame
    3. Normalise country names across sources
    4. Union into one long Spark DF
    5. Pivot to wide Spark DF (one row per country + year)
    6. Drop rows missing both GDP and HDI

    Args:
        include_api: pass True to also fetch live World Bank data.

    Returns:
        (long_df, wide_df) — both are Spark DataFrames, wide_df is cleaned.
    """
    from etl.extract import extract_all  # imported here to keep Spark startup lazy

    spark = get_spark()

    print("=== Extract ===")
    pdfs = extract_all(include_api=include_api)

    print("\n=== Pandas → Spark ===")
    spark_dfs = to_spark_dict(pdfs, spark)

    print("\n=== Normalise country names ===")
    long_df = union_sources(spark_dfs)
    long_df = normalize_country_names(long_df)

    print("\n=== Pivot wide (merge on country + year) ===")
    wide_df = pivot_wide(long_df)

    print("\n=== Clean ===")
    wide_df = drop_missing_gdp_hdi(wide_df)

    print("\n=== Done ===")
    return long_df, wide_df
