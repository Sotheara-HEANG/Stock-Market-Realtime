"""
load.py — three-layer warehouse write: Bronze → Silver → Gold.

Bronze  (bronze.raw_finance)    — raw long-format records from extract.py
Silver  (silver.finance_prices) — cleaned wide records from transform + enrich
Gold    (gold.dim_asset,        — dimensional model for analytics
         gold.fact_prices)

predict.py separately writes gold.fact_predictions after forecasting.

Usage:
    from etl.load import load
    load(raw_pdfs, enriched_df)   # reads DB credentials from .env
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import DataFrame
from sqlalchemy import create_engine, text, exc, Float
from sqlalchemy.dialects.oracle import FLOAT as OracleFloat

_ENV_PATH = Path(__file__).parent.parent / ".env"

# Price columns written to Silver and Gold fact_prices
_PRICE_COLS = [
    "current_price_usd",
    "open_price_usd",
    "day_high_usd",
    "day_low_usd",
    "previous_close_usd",
    "price_change_usd",
    "price_change_pct",
    "trading_volume",
    "intraday_range_pct",
    "price_momentum",
    "sector_avg_price",
    "sector_avg_change_pct",
]


# ---------------------------------------------------------------------------
# DB connections
# ---------------------------------------------------------------------------

def _get_engine():
    load_dotenv(_ENV_PATH)
    host     = os.environ["DB_HOST"]
    port     = os.environ.get("DB_PORT", "5432")
    dbname   = os.environ["DB_NAME"]
    user     = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


def _get_oracle_engine():
    load_dotenv(_ENV_PATH)
    host     = os.environ.get("ORA_HOST", "localhost")
    port     = os.environ.get("ORA_PORT", "1521")
    service  = os.environ.get("ORA_SERVICE", "FREEPDB1")
    user     = os.environ["ORA_USER"]
    password = os.environ["ORA_PASSWORD"]
    url = f"oracle+oracledb://{user}:{password}@{host}:{port}/?service_name={service}"
    return create_engine(url, pool_pre_ping=True)


def _oracle_float_dtypes(df: pd.DataFrame) -> dict:
    return {
        col: Float().with_variant(OracleFloat(binary_precision=126), "oracle")
        for col in df.select_dtypes(include=["float64", "float32"]).columns
    }


# ---------------------------------------------------------------------------
# Bronze — raw long-format from extract.py
# ---------------------------------------------------------------------------

def _write_bronze(raw_pdfs: dict[str, pd.DataFrame], engine, chunksize: int) -> None:
    """Append all raw extract records to bronze.raw_finance."""
    frames = [df for df in raw_pdfs.values() if len(df) > 0]
    if not frames:
        print("  Bronze: no raw data to write")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"country_name": "company_name"})
    combined["ingested_at"] = datetime.datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))
        conn.execute(text("DROP TABLE IF EXISTS bronze.raw_finance CASCADE"))
        conn.execute(text("""
            CREATE TABLE bronze.raw_finance (
                id           SERIAL       PRIMARY KEY,
                company_name VARCHAR(200) NOT NULL,
                country_code VARCHAR(10),
                indicator    VARCHAR(100) NOT NULL,
                year         INT          NOT NULL,
                value        FLOAT,
                source       VARCHAR(100),
                ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """))

    out_cols = ["company_name", "country_code", "indicator", "year", "value", "source", "ingested_at"]
    available = [c for c in out_cols if c in combined.columns]
    combined[available].to_sql(
        "raw_finance", engine, schema="bronze",
        if_exists="append", index=False, chunksize=chunksize,
    )
    print(f"  Bronze: {len(combined):,} rows → bronze.raw_finance")


# ---------------------------------------------------------------------------
# Silver — cleaned wide-format from transform + enrich
# ---------------------------------------------------------------------------

def _write_silver(pdf: pd.DataFrame, engine, chunksize: int) -> None:
    """Write cleaned enriched rows to silver.finance_prices."""
    silver_cols = ["country_name", "sector", "year"] + _PRICE_COLS
    available   = [c for c in silver_cols if c in pdf.columns]

    silver_df = pdf[available].copy().rename(columns={"country_name": "company_name"})
    silver_df["loaded_at"] = datetime.datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver"))
        conn.execute(text("DROP TABLE IF EXISTS silver.finance_prices CASCADE"))

    silver_df.to_sql(
        "finance_prices", engine, schema="silver",
        if_exists="replace", index=False, chunksize=chunksize,
    )
    print(f"  Silver: {len(silver_df):,} rows → silver.finance_prices")


# ---------------------------------------------------------------------------
# Gold — dimensional model
# ---------------------------------------------------------------------------

def _build_dim_asset(pdf: pd.DataFrame) -> pd.DataFrame:
    cols = ["country_name"]
    if "sector" in pdf.columns:
        cols.append("sector")
    dim = (
        pdf[cols]
        .drop_duplicates(subset=["country_name"])
        .rename(columns={"country_name": "company_name"})
        .reset_index(drop=True)
    )
    dim.insert(0, "asset_id", dim.index + 1)
    if "sector" not in dim.columns:
        dim["sector"] = None
    return dim[["asset_id", "company_name", "sector"]]


def _build_fact_prices(pdf: pd.DataFrame, dim_asset: pd.DataFrame) -> pd.DataFrame:
    price_cols = [c for c in _PRICE_COLS if c in pdf.columns]
    fact = pdf[["country_name", "year"] + price_cols].copy()

    id_map = dim_asset.set_index("company_name")["asset_id"]
    fact["asset_id"] = fact["country_name"].map(id_map)
    fact = fact.dropna(subset=["asset_id"])
    fact["asset_id"] = fact["asset_id"].astype(int)
    fact["loaded_at"] = datetime.datetime.utcnow()

    return fact[["asset_id", "year"] + price_cols + ["loaded_at"]].reset_index(drop=True)


def _write_gold(pdf: pd.DataFrame, engine, chunksize: int) -> None:
    """Write dim_asset and fact_prices to the gold schema."""
    dim_asset   = _build_dim_asset(pdf)
    fact_prices = _build_fact_prices(pdf, dim_asset)

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
        conn.execute(text("DROP TABLE IF EXISTS gold.fact_predictions CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS gold.fact_prices      CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS gold.dim_asset        CASCADE"))
        conn.execute(text("""
            CREATE TABLE gold.dim_asset (
                asset_id     SERIAL       PRIMARY KEY,
                company_name VARCHAR(200) NOT NULL UNIQUE,
                sector       VARCHAR(100),
                created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """))

    dim_asset.to_sql(
        "dim_asset", engine, schema="gold",
        if_exists="append", index=False, chunksize=chunksize,
    )
    print(f"  Gold dim_asset:   {len(dim_asset):,} rows → gold.dim_asset")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE gold.fact_prices (
                id                    SERIAL  PRIMARY KEY,
                asset_id              INT     NOT NULL REFERENCES gold.dim_asset (asset_id),
                year                  INT     NOT NULL,
                current_price_usd     FLOAT,
                open_price_usd        FLOAT,
                day_high_usd          FLOAT,
                day_low_usd           FLOAT,
                previous_close_usd    FLOAT,
                price_change_usd      FLOAT,
                price_change_pct      FLOAT,
                trading_volume        FLOAT,
                intraday_range_pct    FLOAT,
                price_momentum        VARCHAR(20),
                sector_avg_price      FLOAT,
                sector_avg_change_pct FLOAT,
                loaded_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (asset_id, year)
            )
        """))

    fact_prices.to_sql(
        "fact_prices", engine, schema="gold",
        if_exists="append", index=False, chunksize=chunksize,
    )
    print(f"  Gold fact_prices: {len(fact_prices):,} rows → gold.fact_prices")


# ---------------------------------------------------------------------------
# Oracle mirror (gold schema only — Bronze/Silver stay PostgreSQL-only)
# ---------------------------------------------------------------------------

def _write_oracle_gold(pdf: pd.DataFrame, chunksize: int) -> None:
    """Mirror dim_asset and fact_prices to Oracle."""
    try:
        engine    = _get_oracle_engine()
        dim_asset = _build_dim_asset(pdf)
        fact      = _build_fact_prices(pdf, dim_asset)

        ora_dim  = _oracle_float_dtypes(dim_asset)
        ora_fact = _oracle_float_dtypes(fact)

        with engine.begin() as conn:
            for tbl in ("FACT_PRICES", "DIM_ASSET"):
                try:
                    conn.execute(text(f"DROP TABLE {tbl} CASCADE CONSTRAINTS"))
                except exc.DatabaseError as e:
                    if "ORA-00942" not in str(e):
                        raise

        dim_asset.to_sql(
            "dim_asset", engine, if_exists="replace", index=False,
            chunksize=chunksize, dtype=ora_dim,
        )
        fact.to_sql(
            "fact_prices", engine, if_exists="replace", index=False,
            chunksize=chunksize, dtype=ora_fact,
        )
        print(f"  Oracle: dim_asset ({len(dim_asset):,}) + fact_prices ({len(fact):,}) written")
        engine.dispose()
    except Exception as e:
        print(f"  [WARNING] Oracle write failed — skipping. Reason: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load(
    raw_pdfs: dict[str, pd.DataFrame],
    enriched_df: DataFrame,
    chunksize: int = 2_000,
) -> None:
    """
    Write all three warehouse layers to PostgreSQL, then mirror Gold to Oracle.

    Args:
        raw_pdfs    : dict returned by extract_all() — used for Bronze
        enriched_df : Spark DataFrame returned by enrich() — used for Silver + Gold
        chunksize   : rows per batch for pandas to_sql
    """
    print("=== Load — Bronze / Silver / Gold ===")

    print("  [1/5] Collecting Spark DataFrame to pandas...")
    pdf = enriched_df.toPandas()
    print(f"        {len(pdf):,} rows  ×  {len(pdf.columns)} columns")

    print("  [2/5] Connecting to PostgreSQL...")
    pg_engine = _get_engine()
    print(f"        connected → {pg_engine.url.database}@{pg_engine.url.host}")

    print("  [3/5] Writing Bronze layer...")
    _write_bronze(raw_pdfs, pg_engine, chunksize)

    print("  [4/5] Writing Silver layer...")
    _write_silver(pdf, pg_engine, chunksize)

    print("  [5/5] Writing Gold layer (PostgreSQL)...")
    _write_gold(pdf, pg_engine, chunksize)
    pg_engine.dispose()

    print("  [+]   Mirroring Gold to Oracle...")
    _write_oracle_gold(pdf, chunksize)

    print("\n  Done — Bronze / Silver / Gold written.")
