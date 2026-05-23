"""
transform.py - convert pandas stock price DataFrames to PySpark and pivot metrics.

Pipeline:
    pandas DFs (extract.py)
        → Spark DFs (pandas_to_spark)
        → unified long Spark DF (union_sources)       # all rows, all indicators
        → wide Spark DF (pivot_wide)                  # one row per entity+timeframe+time_index, indicators as columns

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
    StructType,
    StructField,
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
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel("WARN")
    return _spark


# ---------------------------------------------------------------------------
# Schema — all sources share this 7-column structure after extract.py
# ---------------------------------------------------------------------------

LONG_SCHEMA = StructType([
    StructField("country_code", StringType(),  nullable=True),
    StructField("country_name", StringType(),  nullable=True),
    StructField("indicator",    StringType(),  nullable=False),
    StructField("timeframe",    StringType(),  nullable=False),
    StructField("time_index",   StringType(),  nullable=False),
    StructField("value",        DoubleType(),  nullable=True),
    StructField("source",       StringType(),  nullable=True),
])


# ---------------------------------------------------------------------------
# Conversion: pandas → Spark
# ---------------------------------------------------------------------------

def pandas_to_spark(pdf: pd.DataFrame, spark: SparkSession) -> DataFrame:
    """
    Convert a single extract.py pandas DataFrame to a Spark DataFrame.
    """
    pdf = pdf.copy()
    pdf["country_code"] = pdf["country_code"].astype(str).replace("nan", "")
    pdf["country_name"] = pdf["country_name"].astype(str).replace("nan", "")
    pdf["indicator"]    = pdf["indicator"].astype(str)
    pdf["timeframe"]    = pdf["timeframe"].astype(str)
    pdf["time_index"]   = pdf["time_index"].astype(str)
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
# Merge step 2: pivot long → wide (one row per country + timeframe + time_index)
# ---------------------------------------------------------------------------

def pivot_wide(long_df: DataFrame) -> DataFrame:
    """
    Pivot the unified long DF to a wide format:
        (country_code, country_name, timeframe, time_index) → one column per indicator
    """
    wide_df = (
        long_df
        .groupBy("country_code", "country_name", "timeframe", "time_index")
        .pivot("indicator")
        .agg(F.first("value"))
    )

    all_cols = sorted(
        set(wide_df.columns),
        key=lambda c: (c not in ("country_code", "country_name", "timeframe", "time_index"), c),
    )
    wide_df = wide_df.select(all_cols)

    print(f"  Wide DF: {wide_df.count():,} rows  ×  {len(wide_df.columns)} columns")
    return wide_df


# Canonical names keyed by ISO-3 / Polity scode.
# Derived from WGI, IMF, Polity5, V-Dem sources for legacy backwards compatibility.
CANONICAL_NAMES: dict[str, str] = {
    "AUS": "Australia",
    "AUT": "Austria",
    "COD": "Democratic Republic of Congo",
    "COG": "Republic of Congo",
    "CON": "Republic of Congo",
    "CPV": "Cabo Verde",
    "CZE": "Czechia",
    "EGY": "Egypt",
    "GMB": "Gambia",
    "GMY": "Germany",
    "HKG": "Hong Kong SAR, China",
    "IRN": "Iran",
    "IVO": "Côte d'Ivoire",
    "CIV": "Côte d'Ivoire",
    "KGZ": "Kyrgyzstan",
    "KOR": "South Korea",
    "PRK": "North Korea",
    "LAO": "Laos",
    "MAC": "Macao SAR, China",
    "MKD": "North Macedonia",
    "PSE": "Palestine",
    "RUS": "Russia",
    "SDN": "Sudan",
    "SLV": "El Salvador",
    "SVN": "Slovenia",
    "SOM": "Somalia",
    "SVK": "Slovakia",
    "SWZ": "Eswatini",
    "CHE": "Switzerland",
    "SYR": "Syria",
    "TLS": "Timor-Leste",
    "TUR": "Turkiye",
    "VEN": "Venezuela",
    "VNM": "Vietnam",
    "YEM": "Yemen",
    "YGS": "Yugoslavia",
    "BHS": "Bahamas",
    "BOL": "Bolivia",
    "BRN": "Brunei",
    "KYR": "Kyrgyzstan",
    "SRB": "Serbia",
}


def normalize_country_names(long_df: DataFrame) -> DataFrame:
    """
    Apply CANONICAL_NAMES to the long DF before pivoting.
    """
    name_map = F.create_map(
        *[x for pair in ((F.lit(k), F.lit(v)) for k, v in CANONICAL_NAMES.items()) for x in pair]
    )
    corrected = long_df.withColumn(
        "country_name",
        F.coalesce(name_map[F.col("country_code")], F.col("country_name")),
    )
    mapped_count = long_df.filter(
        F.col("country_code").isin(list(CANONICAL_NAMES.keys()))
    ).count()
    print(f"  Country name normalisation: {mapped_count:,} rows updated to canonical names")
    return corrected


# ---------------------------------------------------------------------------
# Clean step: drop rows missing prices
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
    return drop_missing_price(wide_df)


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
    """
    from etl.extract import extract_all

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

    print("\n=== Pivot wide (merge on ticker + timeframe + time_index) ===")
    wide_df = pivot_wide(long_df)

    print("\n=== Clean ===")
    wide_df = drop_missing_price(wide_df)

    print("\n=== Done ===")
    return long_df, wide_df
