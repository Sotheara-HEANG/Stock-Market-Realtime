"""
load.py - three-layer warehouse write for Finnhub stock price data.

Bronze  (bronze.raw_commodity_prices) - raw long-format metric records
Silver  (silver.commodity_prices)     - cleaned wide records
Gold    (gold.dim_commodity,
         gold.fact_commodity_prices)
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import DataFrame
from sqlalchemy import create_engine, text

_ENV_PATH = Path(__file__).parent.parent / ".env"

PRICE_METRIC_COLS = [
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "latest_price",
]

ENRICHED_COLS = [
    "commodity_category",
    "price_change",
    "price_change_pct",
    "price_trend",
    "intraday_range",
    "intraday_range_pct",
    "volatility_level",
    "category_avg_close",
    "category_count",
]


def _get_engine():
    load_dotenv(_ENV_PATH)
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


def _normalise_commodity_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
    df = pdf.copy()
    if "country_code" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"country_code": "symbol"})
    if "country_name" in df.columns and "commodity_name" not in df.columns:
        df = df.rename(columns={"country_name": "commodity_name"})
    if "commodity_category" not in df.columns:
        df["commodity_category"] = "Other"
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["commodity_name"] = df["commodity_name"].astype(str)
    df["commodity_category"] = df["commodity_category"].astype(str)
    return df


def _write_bronze(raw_pdfs: dict[str, pd.DataFrame], engine, chunksize: int) -> None:
    """Append all raw stock records to bronze.raw_commodity_prices."""
    frames = [df for df in raw_pdfs.values() if len(df) > 0]
    if not frames:
        print("  Bronze: no raw data to write")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"country_code": "symbol", "country_name": "commodity_name"})
    combined["ingested_at"] = datetime.datetime.utcnow()
    combined["symbol"] = combined["symbol"].astype(str).str.upper()

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))
        conn.execute(text("DROP TABLE IF EXISTS bronze.raw_commodity_prices CASCADE"))
        conn.execute(text("""
            CREATE TABLE bronze.raw_commodity_prices (
                id             SERIAL       PRIMARY KEY,
                symbol         VARCHAR(40)  NOT NULL,
                commodity_name VARCHAR(160),
                indicator      VARCHAR(100) NOT NULL,
                year           INT          NOT NULL,
                value          FLOAT,
                source         VARCHAR(100),
                ingested_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """))

    out_cols = ["symbol", "commodity_name", "indicator", "year", "value", "source", "ingested_at"]
    combined[out_cols].to_sql(
        "raw_commodity_prices", engine, schema="bronze",
        if_exists="append", index=False, chunksize=chunksize,
    )
    print(f"  Bronze: {len(combined):,} rows -> bronze.raw_commodity_prices")


def _write_silver(pdf: pd.DataFrame, engine, chunksize: int) -> None:
    """Write cleaned stock rows to silver.commodity_prices."""
    pdf = _normalise_commodity_pdf(pdf)
    silver_cols = ["symbol", "commodity_name", "year"] + PRICE_METRIC_COLS + ENRICHED_COLS
    available = [col for col in silver_cols if col in pdf.columns]
    silver_df = pdf[available].copy()
    silver_df["loaded_at"] = datetime.datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver"))
        conn.execute(text("DROP TABLE IF EXISTS silver.commodity_prices CASCADE"))

    silver_df.to_sql(
        "commodity_prices", engine, schema="silver",
        if_exists="replace", index=False, chunksize=chunksize,
    )
    print(f"  Silver: {len(silver_df):,} rows -> silver.commodity_prices")


def _build_dim_commodity(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = _normalise_commodity_pdf(pdf)
    dim = (
        pdf[["symbol", "commodity_name", "commodity_category"]]
        .drop_duplicates(subset=["symbol"])
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    dim.insert(0, "commodity_id", dim.index + 1)
    return dim[["commodity_id", "symbol", "commodity_name", "commodity_category"]]


def _build_fact_commodity_prices(pdf: pd.DataFrame, dim_commodity: pd.DataFrame) -> pd.DataFrame:
    pdf = _normalise_commodity_pdf(pdf)
    fact_cols = [col for col in PRICE_METRIC_COLS + ENRICHED_COLS if col in pdf.columns and col != "commodity_category"]
    fact = pdf[["symbol", "year"] + fact_cols].copy()

    id_map = dim_commodity.set_index("symbol")["commodity_id"]
    fact["commodity_id"] = fact["symbol"].map(id_map)
    fact = fact.dropna(subset=["commodity_id"])
    fact["commodity_id"] = fact["commodity_id"].astype(int)
    fact["loaded_at"] = datetime.datetime.utcnow()

    return fact[["commodity_id", "year"] + fact_cols + ["loaded_at"]].reset_index(drop=True)


def _write_gold(pdf: pd.DataFrame, engine, chunksize: int) -> None:
    """Write ticker dimension and price facts to the gold schema."""
    dim_commodity = _build_dim_commodity(pdf)
    fact_prices = _build_fact_commodity_prices(pdf, dim_commodity)

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
        conn.execute(text("DROP TABLE IF EXISTS gold.fact_predictions        CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS gold.fact_commodity_prices   CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS gold.dim_commodity           CASCADE"))
        conn.execute(text("""
            CREATE TABLE gold.dim_commodity (
                commodity_id       SERIAL       PRIMARY KEY,
                symbol             VARCHAR(40)  NOT NULL UNIQUE,
                commodity_name     VARCHAR(160),
                commodity_category VARCHAR(80),
                created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """))

    dim_commodity.to_sql(
        "dim_commodity", engine, schema="gold",
        if_exists="append", index=False, chunksize=chunksize,
    )
    print(f"  Gold dim_commodity:        {len(dim_commodity):,} rows -> gold.dim_commodity")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE gold.fact_commodity_prices (
                id                  SERIAL PRIMARY KEY,
                commodity_id        INT    NOT NULL REFERENCES gold.dim_commodity (commodity_id),
                year                INT    NOT NULL,
                open_price          FLOAT,
                high_price          FLOAT,
                low_price           FLOAT,
                close_price         FLOAT,
                latest_price        FLOAT,
                price_change        FLOAT,
                price_change_pct    FLOAT,
                price_trend         VARCHAR(20),
                intraday_range      FLOAT,
                intraday_range_pct  FLOAT,
                volatility_level    VARCHAR(20),
                category_avg_close  FLOAT,
                category_count      INT,
                loaded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (commodity_id, year)
            )
        """))

    fact_prices.to_sql(
        "fact_commodity_prices", engine, schema="gold",
        if_exists="append", index=False, chunksize=chunksize,
    )
    print(f"  Gold fact_commodity_prices:{len(fact_prices):,} rows -> gold.fact_commodity_prices")


def load(
    raw_pdfs: dict[str, pd.DataFrame],
    enriched_df: DataFrame,
    chunksize: int = 2_000,
) -> None:
    """Write all three warehouse layers to PostgreSQL."""
    print("=== Load - Bronze / Silver / Gold ===")

    print("  [1/5] Collecting Spark DataFrame to pandas...")
    pdf = enriched_df.toPandas()
    print(f"        {len(pdf):,} rows x {len(pdf.columns)} columns")

    print("  [2/5] Connecting to PostgreSQL...")
    pg_engine = _get_engine()
    print(f"        connected -> {pg_engine.url.database}@{pg_engine.url.host}")

    print("  [3/5] Writing Bronze layer...")
    _write_bronze(raw_pdfs, pg_engine, chunksize)

    print("  [4/5] Writing Silver layer...")
    _write_silver(pdf, pg_engine, chunksize)

    print("  [5/5] Writing Gold layer (PostgreSQL)...")
    _write_gold(pdf, pg_engine, chunksize)
    pg_engine.dispose()

    print("\n  Done - Bronze / Silver / Gold written.")
