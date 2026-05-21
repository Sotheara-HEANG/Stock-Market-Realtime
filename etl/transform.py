"""
transform.py - convert pandas stock price DataFrames to PySpark and pivot metrics.

Pipeline:
    pandas DFs (extract.py)
        → Spark DFs (pandas_to_spark)
        → unified long Spark DF (union_sources)       # all rows, all indicators
        → wide Spark DF (pivot_wide)                  # one row per entity+year, indicators as columns

Usage:
    from etl.transform import transform

    long_df, wide_df = transform()
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
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


def _parse_java_major(version_output: str) -> int | None:
    match = re.search(r'version "([^"]+)"', version_output)
    if not match:
        return None

    raw_version = match.group(1).split(".", maxsplit=2)
    try:
        if raw_version[0] == "1" and len(raw_version) > 1:
            return int(raw_version[1])
        return int(raw_version[0])
    except ValueError:
        return None


def _java_major(java_bin: str = "java") -> int | None:
    try:
        proc = subprocess.run(
            [java_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_java_major(proc.stderr + proc.stdout)


def _candidate_java_homes() -> list[Path]:
    homes: list[Path] = []

    configured = os.environ.get("JAVA_HOME")
    if configured:
        homes.append(Path(configured))

    for root in (Path("/Library/Java/JavaVirtualMachines"), Path("/usr/lib/jvm")):
        if root.exists():
            homes.extend(path / "Contents" / "Home" for path in root.glob("*.jdk"))
            homes.extend(path for path in root.glob("*") if (path / "bin" / "java").exists())

    homebrew_root = Path("/opt/homebrew/opt")
    if homebrew_root.exists():
        homes.extend(homebrew_root.glob("openjdk*/libexec/openjdk.jdk/Contents/Home"))

    unique: list[Path] = []
    seen: set[str] = set()
    for home in homes:
        resolved = str(home)
        if resolved not in seen and (home / "bin" / "java").exists():
            unique.append(home)
            seen.add(resolved)
    return unique


def _ensure_supported_java(min_major: int = 17) -> None:
    current_major = _java_major()
    if current_major is not None and current_major >= min_major:
        return

    compatible: list[tuple[int, Path]] = []
    for home in _candidate_java_homes():
        major = _java_major(str(home / "bin" / "java"))
        if major is not None and major >= min_major:
            compatible.append((major, home))

    if compatible:
        _, java_home = sorted(compatible, key=lambda item: item[0])[0]
        os.environ["JAVA_HOME"] = str(java_home)
        os.environ["PATH"] = f"{java_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"  Using Java {java_home} for PySpark")
        return

    detected = f"Java {current_major}" if current_major is not None else "no Java runtime"
    raise RuntimeError(
        f"PySpark requires Java {min_major}+ but found {detected}. "
        "Install JDK 17+ or set JAVA_HOME to a compatible JDK before running the pipeline."
    )


def get_spark() -> SparkSession:
    """Return a singleton SparkSession, creating it on first call."""
    global _spark
    if _spark is None:
        _ensure_supported_java()
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        _spark = (
            SparkSession.builder
            .appName("finnhub-trading-pipeline")
            .config("spark.sql.session.timeZone", "UTC")
            # Reduce default shuffle partitions for local single-node runs
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
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
        (country_code, country_name, year) → one column per indicator

    For the trading pipeline, country_code carries the ticker and country_name
    carries the company display name.
    """
    wide_df = (
        long_df
        .groupBy("country_code", "country_name", "year")
        .pivot("indicator")
        .agg(F.first("value"))
    )

    all_cols = sorted(
        set(wide_df.columns),
        key=lambda c: (c not in ("country_code", "country_name", "year"), c),
    )
    wide_df = wide_df.select(all_cols)

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

def drop_missing_price(wide_df: DataFrame) -> DataFrame:
    """
    Drop rows where core price metrics are missing.
    """
    needed = {"close_price"}
    available = needed.intersection(wide_df.columns)
    if not available:
        print("  No close_price column - no rows dropped")
        return wide_df
    before = wide_df.count()
    cleaned = wide_df
    for col_name in available:
        cleaned = cleaned.filter(F.col(col_name).isNotNull())
    after = cleaned.count()
    print(f"  Dropped {before - after:,} rows missing prices ({after:,} remain)")
    return cleaned


def drop_missing_email_score(wide_df: DataFrame) -> DataFrame:
    """Backward-compatible alias for older callers."""
    return drop_missing_price(wide_df)


def drop_missing_gdp_hdi(wide_df: DataFrame) -> DataFrame:
    """Legacy filter for country-based data — kept for backward compatibility."""
    if "gdp_growth_pct" not in wide_df.columns and "hdi_value" not in wide_df.columns:
        return drop_missing_price(wide_df)
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

def transform(
    include_realtime: bool = True,
    include_legacy: bool = False,
    include_api: bool = False,
) -> tuple[DataFrame, DataFrame]:
    """
    Run the full transform + clean pipeline.

    1. Extract sources via extract.py (pandas)
    2. Convert each to a Spark DataFrame
    3. Union into one long Spark DF
    4. Pivot to wide Spark DF (one row per ticker + year)
    5. Drop rows missing core price metrics

    Args:
        include_realtime: fetch Finnhub stock data (default True).
        include_legacy:   kept for CLI compatibility; ignored by active extractor.
        include_api:      kept for CLI compatibility; ignored by active extractor.

    Returns:
        (long_df, wide_df) — both are Spark DataFrames, wide_df is cleaned.
    """
    from etl.extract import extract_all  # imported here to keep Spark startup lazy

    spark = get_spark()

    print("=== Extract ===")
    pdfs = extract_all(
        include_realtime=include_realtime,
        include_legacy=include_legacy,
        include_api=include_api,
    )

    print("\n=== Pandas → Spark ===")
    spark_dfs = to_spark_dict(pdfs, spark)

    print("\n=== Union sources ===")
    long_df = union_sources(spark_dfs)

    if include_legacy:
        print("\n=== Normalise country names ===")
        long_df = normalize_country_names(long_df)

    print("\n=== Pivot wide (merge on ticker + year) ===")
    wide_df = pivot_wide(long_df)

    print("\n=== Clean ===")
    wide_df = drop_missing_price(wide_df)

    print("\n=== Done ===")
    return long_df, wide_df
