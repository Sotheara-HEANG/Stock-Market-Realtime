"""
load.py — write the enriched master DataFrame into PostgreSQL and Oracle (dual-write).

Tables written (in order, respecting FK):
    1. countries  — one row per unique iso_code
    2. indicators — long-format rows keyed by country_id

Uses if_exists='replace' so every run wipes and rewrites both tables cleanly.
Switch to ON CONFLICT upserts once the schema is stable.

Usage:
    from etl.load import load
    load(enriched_df)           # reads DB credentials from .env
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import DataFrame
from sqlalchemy import create_engine, text, exc, Float
from sqlalchemy.dialects.oracle import FLOAT as OracleFloat

_ENV_PATH = Path(__file__).parent.parent / ".env"

# ---------------------------------------------------------------------------
# Indicator metadata: column_name → (source, unit)
# Only columns listed here are written to the indicators table.
# ---------------------------------------------------------------------------

_INDICATOR_META: dict[str, tuple[str, str]] = {
    # RapidAPI real-time finance data
    "current_price_usd":        ("RapidAPI", "USD"),
    "open_price_usd":           ("RapidAPI", "USD"),
    "day_high_usd":             ("RapidAPI", "USD"),
    "day_low_usd":              ("RapidAPI", "USD"),
    "trading_volume":           ("RapidAPI", "count"),
    "previous_close_usd":       ("RapidAPI", "USD"),
    "price_change_usd":         ("RapidAPI", "USD"),
    "price_change_pct":         ("RapidAPI", "percent"),
    # Derived features (enrich.py)
    "intraday_range_pct":       ("derived",  "percent"),
    "sector_avg_price":         ("derived",  "USD"),
    "sector_avg_change_pct":    ("derived",  "percent"),
}

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------


def _get_engine():
    """Build a SQLAlchemy engine for PostgreSQL from .env."""
    load_dotenv(_ENV_PATH)
    host     = os.environ["DB_HOST"]
    port     = os.environ.get("DB_PORT", "5432")
    dbname   = os.environ["DB_NAME"]
    user     = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


def _get_oracle_engine():
    """Build a SQLAlchemy engine for Oracle (Docker) from .env."""
    load_dotenv(_ENV_PATH)
    host    = os.environ.get("ORA_HOST", "localhost")
    port    = os.environ.get("ORA_PORT", "1521")
    service = os.environ.get("ORA_SERVICE", "FREEPDB1")
    user    = os.environ["ORA_USER"]
    password = os.environ["ORA_PASSWORD"]
    url = f"oracle+oracledb://{user}:{password}@{host}:{port}/?service_name={service}"
    return create_engine(url, pool_pre_ping=True)


def _drop_oracle_tables(conn) -> None:
    """Drop tables in FK-safe order, ignoring ORA-00942 (table does not exist)."""
    for table in ("predictions", "indicators", "countries"):
        try:
            conn.execute(text(f"DROP TABLE {table} CASCADE CONSTRAINTS"))
        except exc.DatabaseError as e:
            if "ORA-00942" not in str(e):
                raise


# ---------------------------------------------------------------------------
# Build tables as pandas DataFrames
# ---------------------------------------------------------------------------


def _build_countries(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Build an assets table keyed by company/asset name (country_name column).

    For real-time finance data, country_code is empty and country_name holds
    the company name (e.g. "Apple Inc"). Each unique name gets a sequential id.
    The sector column is optional — present only when enrich() has been run.
    """
    cols = ["country_name"]
    if "sector" in pdf.columns:
        cols.append("sector")

    companies = (
        pdf[cols]
        .drop_duplicates(subset=["country_name"])
        .rename(columns={"country_name": "name", "sector": "region"})
        .reset_index(drop=True)
    )

    companies["iso_code"] = ""
    if "region" not in companies.columns:
        companies["region"] = None

    companies.insert(0, "id", companies.index + 1)
    return companies[["id", "iso_code", "name", "region"]]


def _build_indicators(pdf: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the wide DataFrame to long format, attach asset_id via the
    countries table (keyed by name), and attach source/unit metadata.

    Rows with null values and rows whose company name has no match are dropped.
    """
    indicator_cols = [c for c in pdf.columns if c in _INDICATOR_META]

    long = pdf[["country_name", "year"] + indicator_cols].melt(
        id_vars=["country_name", "year"],
        value_vars=indicator_cols,
        var_name="indicator",
        value_name="value",
    )

    long = long.dropna(subset=["value", "country_name"])
    long = long[long["country_name"].str.strip() != ""]

    long["source"] = long["indicator"].map(lambda c: _INDICATOR_META[c][0])
    long["unit"]   = long["indicator"].map(lambda c: _INDICATOR_META[c][1])
    long["year"]   = long["year"].astype(int)

    # Join to get company id; rows with no match are dropped
    id_map = countries.set_index("name")["id"]
    long["country_id"] = long["country_name"].map(id_map)
    long = long.dropna(subset=["country_id"])
    long["country_id"] = long["country_id"].astype(int)

    return (
        long[["country_id", "indicator", "source", "year", "value", "unit"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _oracle_float_dtypes(df: pd.DataFrame) -> dict:
    """
    Return a dtype mapping for float64 columns using Oracle's FLOAT(binary_precision=126).
    Without this, SQLAlchemy raises an error about decimal vs binary precision mismatch.
    binary_precision=126 is the Oracle equivalent of IEEE 754 double (float64).
    """
    return {
        col: Float().with_variant(OracleFloat(binary_precision=126), "oracle")
        for col in df.select_dtypes(include=["float64", "float32"]).columns
    }


def _write_tables(engine, countries: pd.DataFrame, indicators: pd.DataFrame, label: str, chunksize: int, is_oracle: bool = False) -> None:
    """Write countries and indicators to the given engine."""
    ora_dtype_c = _oracle_float_dtypes(countries)  if is_oracle else None
    ora_dtype_i = _oracle_float_dtypes(indicators) if is_oracle else None

    print(f"  Writing countries → {label}...")
    if is_oracle:
        with engine.begin() as conn:
            _drop_oracle_tables(conn)
        countries.to_sql("countries", engine, if_exists="replace", index=False, chunksize=chunksize, dtype=ora_dtype_c)
    else:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS predictions CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS indicators  CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS countries   CASCADE"))
        countries.to_sql("countries", engine, if_exists="replace", index=False, chunksize=chunksize)
    print(f"        {len(countries):,} rows written")

    print(f"  Writing indicators → {label}...")
    indicators.to_sql("indicators", engine, if_exists="replace", index=False, chunksize=chunksize, dtype=ora_dtype_i)
    print(f"        {len(indicators):,} rows written")

    engine.dispose()


def load(enriched_df: DataFrame, chunksize: int = 2_000) -> None:
    """
    Dual-write the enriched master DataFrame to PostgreSQL and Oracle.

    Args:
        enriched_df : Spark DataFrame returned by enrich.enrich()
        chunksize   : rows per batch passed to pandas to_sql
    """
    print("=== Load (dual-write: PostgreSQL + Oracle) ===")

    print("  [1/5] Collecting Spark DataFrame to pandas...")
    pdf = enriched_df.toPandas()
    print(f"        {len(pdf):,} rows  ×  {len(pdf.columns)} columns")

    print("  [2/5] Building tables...")
    countries  = _build_countries(pdf)
    indicators = _build_indicators(pdf, countries)

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    print("  [3/5] Connecting to PostgreSQL...")
    pg_engine = _get_engine()
    print(f"        connected  →  {pg_engine.url.database}@{pg_engine.url.host}")
    _write_tables(pg_engine, countries, indicators, label="PostgreSQL", chunksize=chunksize)

    # ── Oracle ──────────────────────────────────────────────────────────────
    print("  [4/5] Connecting to Oracle (Docker)...")
    try:
        ora_engine = _get_oracle_engine()
        print(f"        connected  →  {ora_engine.url.host}:{ora_engine.url.port}")
        _write_tables(ora_engine, countries, indicators, label="Oracle", chunksize=chunksize, is_oracle=True)
    except Exception as e:
        print(f"  [WARNING] Oracle write failed — skipping. Reason: {e}")
        print("            PostgreSQL write succeeded; Oracle data may be out of sync.")

    print("  [5/5] Done.")
