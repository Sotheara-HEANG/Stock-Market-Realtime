"""
spark_db_extract.py — extract data from PostgreSQL/MySQL/SQL Server into Spark via JDBC.

Reads the 'countries' and 'indicators' tables, joins them, and returns a
denormalized long-format Spark DataFrame ready for EDA and aggregation.

Supported db_type values:
    'postgresql'  — default (port 5432)
    'mysql'       — MySQL / MariaDB (port 3306)
    'sqlserver'   — Microsoft SQL Server (port 1433)

The JDBC driver JAR is downloaded automatically from Maven Central on first run
and cached in ~/.ivy2/. Subsequent runs are fully offline.

Usage:
    from etl.spark_db_extract import get_spark_jdbc, extract_from_db

    spark = get_spark_jdbc('postgresql')
    df    = extract_from_db(spark, db_type='postgresql')
    df.show(5)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

_ENV_PATH = Path(__file__).parent.parent / ".env"

_DEFAULT_PORTS = {
    "postgresql": "5432",
    "mysql":      "3306",
    "sqlserver":  "1433",
}

_JDBC_PACKAGES = {
    "postgresql": "org.postgresql:postgresql:42.7.3",
    "mysql":      "com.mysql:mysql-connector-j:8.3.0",
    "sqlserver":  "com.microsoft.sqlserver:mssql-jdbc:12.4.2.jre11",
}

_JDBC_DRIVERS = {
    "postgresql": "org.postgresql.Driver",
    "mysql":      "com.mysql.cj.jdbc.Driver",
    "sqlserver":  "com.microsoft.sqlserver.jdbc.SQLServerDriver",
}


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _build_jdbc_url(db_type: str) -> tuple[str, dict[str, str]]:
    """Return (jdbc_url, connection_properties) built from .env variables."""
    load_dotenv(_ENV_PATH)

    host     = os.environ["DB_HOST"]
    port     = os.environ.get("DB_PORT", _DEFAULT_PORTS[db_type])
    dbname   = os.environ["DB_NAME"]
    user     = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")

    if db_type == "postgresql":
        url = f"jdbc:postgresql://{host}:{port}/{dbname}"
    elif db_type == "mysql":
        url = (
            f"jdbc:mysql://{host}:{port}/{dbname}"
            "?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC"
        )
    elif db_type == "sqlserver":
        url = (
            f"jdbc:sqlserver://{host}:{port};"
            f"databaseName={dbname};encrypt=true;trustServerCertificate=true"
        )
    else:
        supported = list(_JDBC_PACKAGES)
        raise ValueError(f"Unsupported db_type {db_type!r}. Choose from: {supported}")

    props = {
        "user":     user,
        "password": password,
        "driver":   _JDBC_DRIVERS[db_type],
    }
    return url, props


# ---------------------------------------------------------------------------
# SparkSession factory
# ---------------------------------------------------------------------------

def get_spark_jdbc(db_type: str = "postgresql") -> SparkSession:
    """
    Create a SparkSession pre-loaded with the JDBC driver for db_type.

    The driver JAR is resolved from Maven Central the first time and cached
    locally. Pass db_type to match your target database.
    """
    package = _JDBC_PACKAGES.get(db_type)
    if not package:
        raise ValueError(f"Unsupported db_type: {db_type!r}")

    spark = (
        SparkSession.builder
        .appName(f"econ-db-analysis-{db_type}")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.jars.packages", package)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Table reader
# ---------------------------------------------------------------------------

def _read_table(
    spark: SparkSession,
    table: str,
    url: str,
    props: dict[str, str],
) -> DataFrame:
    return (
        spark.read.format("jdbc")
        .option("url", url)
        .option("dbtable", table)
        .option("user", props["user"])
        .option("password", props["password"])
        .option("driver", props["driver"])
        .load()
    )


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_from_db(
    spark: SparkSession,
    db_type: str = "postgresql",
) -> DataFrame:
    """
    Read 'countries' + 'indicators' tables and return a joined Spark DataFrame.

    Output schema (one row per country × indicator × year):
        country_id   int
        iso_code     str   — ISO-3 country code
        country_name str
        region       str   — continent / region from the countries table
        indicator    str   — snake_case indicator name
        source       str   — originating dataset (WGI, IMF, …)
        year         int
        value        double
        unit         str

    Args:
        spark  : SparkSession created by get_spark_jdbc(db_type)
        db_type: one of 'postgresql', 'mysql', 'sqlserver'

    Returns:
        Spark DataFrame, filtered to non-null values only.
    """
    url, props = _build_jdbc_url(db_type)
    host_display = url.split("//")[-1].split("/")[0]
    print(f"[extract_from_db] Connecting to {db_type} at {host_display}...")

    countries  = _read_table(spark, "countries",  url, props)
    indicators = _read_table(spark, "indicators", url, props)

    n_countries  = countries.count()
    n_indicators = indicators.count()
    print(f"  countries  table: {n_countries:,} rows")
    print(f"  indicators table: {n_indicators:,} rows")

    df = (
        indicators.join(
            countries.select(
                F.col("id").alias("country_id"),
                F.col("iso_code"),
                F.col("name").alias("country_name"),
                F.col("region"),
            ),
            on="country_id",
            how="left",
        )
        .select(
            "country_id",
            "iso_code",
            "country_name",
            "region",
            "indicator",
            "source",
            F.col("year").cast("int").alias("year"),
            F.col("value").cast("double").alias("value"),
            "unit",
        )
        .filter(F.col("value").isNotNull())
        .cache()
    )

    total = df.count()
    print(f"  Joined + filtered: {total:,} rows  ×  {len(df.columns)} columns")
    return df
