"""
app.py - FastAPI backend for the Finnhub stock market pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from etl.extract import extract_latest_stock_prices

_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)

app = FastAPI(
    title="Finnhub Trading API",
    description="Serves stock prices, warehouse analytics, and forecasts.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _engine():
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    engine = _engine()
    try:
        return pd.read_sql(text(sql), engine, params=params)
    finally:
        engine.dispose()


def _records(df: pd.DataFrame) -> list[dict]:
    """Return JSON-safe records with pandas NaN/NaT converted to None."""
    if df.empty:
        return []
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def _record(row: pd.Series) -> dict:
    return row.astype(object).where(pd.notnull(row), None).to_dict()


def _tables_exist(*table_names: str) -> bool:
    engine = _engine()
    try:
        with engine.connect() as conn:
            for table_name in table_names:
                exists = conn.execute(
                    text("SELECT to_regclass(:table_name) IS NOT NULL"),
                    {"table_name": table_name},
                ).scalar_one()
                if not exists:
                    return False
            return True
    finally:
        engine.dispose()


def _live_quote(symbol: str) -> dict:
    df = extract_latest_stock_prices([symbol.upper()])
    if df.empty:
        raise HTTPException(404, f"No stock quote returned for '{symbol}'")

    wide = (
        df.pivot_table(
            index=["country_name", "country_code", "year", "source"],
            columns="indicator",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={
            "country_name": "commodity_name",
            "country_code": "symbol",
        })
    )
    return _record(wide.iloc[0])


@app.get("/health")
def health():
    """Health check - confirms DB connectivity."""
    try:
        _query("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")


@app.get("/quote")
def quote(symbol: str = Query(..., description="Stock ticker, for example AAPL")):
    """Run a live Finnhub quote request."""
    return _live_quote(symbol)


@app.get("/commodities")
def list_commodities():
    """List all tracked stocks from gold.dim_commodity."""
    if not _tables_exist("gold.dim_commodity"):
        return []

    df = _query("""
        SELECT commodity_id, symbol, commodity_name, commodity_category
        FROM   gold.dim_commodity
        ORDER  BY symbol
    """)
    return _records(df)


@app.get("/prices")
def list_prices(
    category: Optional[str] = None,
    trend: Optional[str] = None,
    volatility: Optional[str] = None,
):
    """Stock prices and enrichment metrics, optionally filtered."""
    if not _tables_exist("gold.dim_commodity", "gold.fact_commodity_prices"):
        return []

    clauses = []
    params: dict[str, str] = {}
    if category:
        clauses.append("d.commodity_category = :category")
        params["category"] = category
    if trend:
        clauses.append("f.price_trend = :trend")
        params["trend"] = trend.lower()
    if volatility:
        clauses.append("f.volatility_level = :volatility")
        params["volatility"] = volatility.lower()

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    df = _query(f"""
        SELECT d.symbol,
               d.commodity_name,
               d.commodity_category,
               f.year,
               f.open_price,
               f.high_price,
               f.low_price,
               f.close_price,
               f.latest_price,
               f.price_change,
               f.price_change_pct,
               f.price_trend,
               f.intraday_range,
               f.intraday_range_pct,
               f.volatility_level,
               f.category_avg_close,
               f.category_count
        FROM   gold.fact_commodity_prices f
        JOIN   gold.dim_commodity         d ON d.commodity_id = f.commodity_id
        {where}
        ORDER  BY f.year DESC, d.symbol
    """, params)
    return _records(df)


@app.get("/prices/{symbol}")
def get_prices(symbol: str):
    """Historical price metrics for a single stock ticker."""
    if not _tables_exist("gold.dim_commodity", "gold.fact_commodity_prices"):
        raise HTTPException(404, "Warehouse has no stock price data yet")

    df = _query("""
        SELECT d.symbol, d.commodity_name, d.commodity_category, f.*
        FROM   gold.fact_commodity_prices f
        JOIN   gold.dim_commodity         d ON d.commodity_id = f.commodity_id
        WHERE  d.symbol = :symbol
        ORDER  BY f.year DESC
    """, {"symbol": symbol.upper()})
    if df.empty:
        raise HTTPException(404, f"Ticker '{symbol}' not found")
    return _records(df)


@app.get("/predictions/{symbol}")
def get_predictions(symbol: str, model: Optional[str] = None):
    """ML forecasts for a single stock ticker."""
    if not _tables_exist("gold.dim_commodity", "gold.fact_predictions"):
        raise HTTPException(404, "Warehouse has no predictions yet")

    where = "AND p.model_name = :model" if model else ""
    params = {"symbol": symbol.upper()}
    if model:
        params["model"] = model

    df = _query(f"""
        SELECT d.symbol,
               d.commodity_name,
               d.commodity_category,
               p.indicator,
               p.model_name,
               p.predicted_year,
               p.predicted_value,
               p.confidence_low,
               p.confidence_high,
               p.run_at
        FROM   gold.fact_predictions p
        JOIN   gold.dim_commodity    d ON d.commodity_id = p.commodity_id
        WHERE  d.symbol = :symbol {where}
        ORDER  BY p.indicator, p.model_name, p.predicted_year
    """, params)
    if df.empty:
        raise HTTPException(404, f"No predictions found for '{symbol}'")
    return _records(df)


@app.get("/categories")
def list_categories():
    """List all market sectors/categories."""
    if not _tables_exist("gold.dim_commodity"):
        return []

    df = _query("""
        SELECT DISTINCT commodity_category
        FROM   gold.dim_commodity
        WHERE  commodity_category IS NOT NULL
        ORDER  BY commodity_category
    """)
    return df["commodity_category"].tolist()


@app.get("/categories/{category}/prices")
def category_prices(category: str):
    """Latest rows for every tracked stock in a sector/category."""
    if not _tables_exist("gold.dim_commodity", "gold.fact_commodity_prices"):
        raise HTTPException(404, "Warehouse has no stock price data yet")

    df = _query("""
        SELECT d.symbol,
               d.commodity_name,
               d.commodity_category,
               f.year,
               f.close_price,
               f.price_change_pct,
               f.price_trend,
               f.volatility_level
        FROM   gold.fact_commodity_prices f
        JOIN   gold.dim_commodity         d ON d.commodity_id = f.commodity_id
        WHERE  d.commodity_category = :category
        ORDER  BY f.year DESC, d.symbol
    """, {"category": category})
    if df.empty:
        raise HTTPException(404, f"No stocks found for category '{category}'")
    return _records(df)


@app.get("/top")
def top_commodities(
    indicator: str = Query("close_price", description="Metric to rank by"),
    n: int = Query(10, description="Number of results"),
    order: str = Query("desc", description="asc or desc"),
):
    """Top N stocks ranked by a numeric indicator."""
    if not _tables_exist("gold.dim_commodity", "gold.fact_commodity_prices"):
        return []

    valid = {
        "open_price", "high_price", "low_price", "close_price", "latest_price",
        "price_change", "price_change_pct", "intraday_range", "intraday_range_pct",
        "category_avg_close",
    }
    if indicator not in valid:
        raise HTTPException(400, f"indicator must be one of {sorted(valid)}")
    direction = "DESC" if order.lower() != "asc" else "ASC"
    df = _query(f"""
        SELECT d.symbol,
               d.commodity_name,
               d.commodity_category,
               f.year,
               f.close_price,
               f.price_change_pct,
               f.price_trend,
               f.volatility_level,
               f.{indicator}
        FROM   gold.fact_commodity_prices f
        JOIN   gold.dim_commodity         d ON d.commodity_id = f.commodity_id
        WHERE  f.{indicator} IS NOT NULL
        ORDER  BY f.{indicator} {direction} NULLS LAST
        LIMIT  :n
    """, {"n": n})
    return _records(df)


@app.get("/compare")
def compare_commodities(
    symbols: str = Query(..., description="Comma-separated stock tickers"),
):
    """Side-by-side comparison for multiple stock tickers."""
    if not _tables_exist("gold.dim_commodity", "gold.fact_commodity_prices"):
        return []

    names = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
    if not names:
        raise HTTPException(400, "Provide at least one stock ticker")

    df = _query("""
        SELECT d.symbol,
               d.commodity_name,
               d.commodity_category,
               f.year,
               f.close_price,
               f.price_change_pct,
               f.price_trend,
               f.volatility_level
        FROM   gold.fact_commodity_prices f
        JOIN   gold.dim_commodity         d ON d.commodity_id = f.commodity_id
        WHERE  d.symbol = ANY(:symbols)
        ORDER  BY d.symbol, f.year DESC
    """, {"symbols": names})
    return _records(df)
