"""
app.py — FastAPI backend for the real-time finance pipeline.

Reads from the Gold warehouse layer (gold.dim_asset, gold.fact_prices,
gold.fact_predictions) and exposes a REST API consumed by the dashboard.

Endpoints:
  GET /health
  GET /assets                              — list all tracked companies
  GET /prices                              — current prices for all assets
  GET /prices/{asset_name}                 — metrics for one asset
  GET /predictions/{asset_name}            — forecasts for one asset
  GET /sectors                             — list distinct sectors
  GET /sectors/{sector}/prices             — prices for all assets in a sector
  GET /top?indicator=price_change_pct&n=10 — top N assets by any indicator
  GET /compare?assets=Apple Inc,NVIDIA Corp— side-by-side comparison

Usage:
    uvicorn 05_app.api.app:app --reload --port 8000
    # or from the project root:
    python -m uvicorn 05_app.api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

app = FastAPI(
    title="Real-Time Finance Pipeline API",
    description="Serves data from the Bronze/Silver/Gold warehouse.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def _engine():
    host     = os.environ["DB_HOST"]
    port     = os.environ.get("DB_PORT", "5432")
    dbname   = os.environ["DB_NAME"]
    user     = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url      = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    engine = _engine()
    try:
        return pd.read_sql(text(sql), engine, params=params)
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — confirms DB connectivity."""
    try:
        _query("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")


@app.get("/assets")
def list_assets():
    """List all tracked companies from gold.dim_asset."""
    df = _query("""
        SELECT asset_id, company_name, sector
        FROM   gold.dim_asset
        ORDER  BY company_name
    """)
    return df.to_dict(orient="records")


@app.get("/prices")
def list_prices(sector: Optional[str] = None):
    """
    Current price metrics for all assets, ordered by price_change_pct desc.
    Optional: filter by ?sector=Technology
    """
    where = "AND d.sector = :sector" if sector else ""
    df = _query(f"""
        SELECT d.company_name,
               d.sector,
               f.year,
               f.current_price_usd,
               f.open_price_usd,
               f.day_high_usd,
               f.day_low_usd,
               f.previous_close_usd,
               f.price_change_usd,
               f.price_change_pct,
               f.trading_volume,
               f.intraday_range_pct,
               f.price_momentum,
               f.sector_avg_price,
               f.sector_avg_change_pct
        FROM   gold.fact_prices f
        JOIN   gold.dim_asset   d ON d.asset_id = f.asset_id
        WHERE  1=1 {where}
        ORDER  BY f.price_change_pct DESC NULLS LAST
    """, {"sector": sector} if sector else None)
    return df.to_dict(orient="records")


@app.get("/prices/{asset_name}")
def get_price(asset_name: str):
    """Price metrics for a single asset."""
    df = _query("""
        SELECT d.company_name, d.sector, f.*
        FROM   gold.fact_prices f
        JOIN   gold.dim_asset   d ON d.asset_id = f.asset_id
        WHERE  d.company_name = :name
        ORDER  BY f.year DESC
    """, {"name": asset_name})
    if df.empty:
        raise HTTPException(404, f"Asset '{asset_name}' not found")
    return df.to_dict(orient="records")


@app.get("/predictions/{asset_name}")
def get_predictions(asset_name: str, model: Optional[str] = None):
    """
    ML forecasts for a single asset.
    Optional: filter by ?model=linear_trend or ?model=holt_smoothing
    """
    where = "AND p.model_name = :model" if model else ""
    df = _query(f"""
        SELECT d.company_name,
               p.indicator,
               p.model_name,
               p.predicted_year,
               p.predicted_value,
               p.confidence_low,
               p.confidence_high,
               p.run_at
        FROM   gold.fact_predictions p
        JOIN   gold.dim_asset        d ON d.asset_id = p.asset_id
        WHERE  d.company_name = :name {where}
        ORDER  BY p.indicator, p.model_name, p.predicted_year
    """, {"name": asset_name, "model": model} if model else {"name": asset_name})
    if df.empty:
        raise HTTPException(404, f"No predictions found for '{asset_name}'")
    return df.to_dict(orient="records")


@app.get("/sectors")
def list_sectors():
    """List all distinct sectors from gold.dim_asset."""
    df = _query("""
        SELECT DISTINCT sector
        FROM   gold.dim_asset
        WHERE  sector IS NOT NULL
        ORDER  BY sector
    """)
    return df["sector"].tolist()


@app.get("/sectors/{sector}/prices")
def sector_prices(sector: str):
    """Current prices for every asset in a sector."""
    df = _query("""
        SELECT d.company_name,
               f.current_price_usd,
               f.price_change_pct,
               f.trading_volume,
               f.intraday_range_pct,
               f.price_momentum
        FROM   gold.fact_prices f
        JOIN   gold.dim_asset   d ON d.asset_id = f.asset_id
        WHERE  d.sector = :sector
        ORDER  BY f.price_change_pct DESC NULLS LAST
    """, {"sector": sector})
    if df.empty:
        raise HTTPException(404, f"No assets in sector '{sector}'")
    return df.to_dict(orient="records")


@app.get("/top")
def top_assets(
    indicator: str = Query("price_change_pct", description="Metric to rank by"),
    n:         int = Query(10,                 description="Number of results"),
    order:     str = Query("desc",             description="asc or desc"),
):
    """Top N assets ranked by any price indicator."""
    valid = {
        "current_price_usd", "price_change_pct",
        "trading_volume", "intraday_range_pct",
        "price_change_usd", "day_high_usd", "day_low_usd",
    }
    if indicator not in valid:
        raise HTTPException(400, f"indicator must be one of {sorted(valid)}")
    direction = "DESC" if order.lower() != "asc" else "ASC"
    df = _query(f"""
        SELECT d.company_name,
               d.sector,
               f.current_price_usd,
               f.price_change_pct,
               f.{indicator}
        FROM   gold.fact_prices f
        JOIN   gold.dim_asset   d ON d.asset_id = f.asset_id
        WHERE  f.{indicator} IS NOT NULL
        ORDER  BY f.{indicator} {direction} NULLS LAST
        LIMIT  :n
    """, {"n": n})
    return df.to_dict(orient="records")


@app.get("/compare")
def compare_assets(
    assets: str = Query(..., description="Comma-separated company names"),
):
    """Side-by-side price comparison for multiple assets."""
    names = [a.strip() for a in assets.split(",") if a.strip()]
    if not names:
        raise HTTPException(400, "Provide at least one asset name")
    placeholders = ", ".join(f"'{n}'" for n in names)
    df = _query(f"""
        SELECT d.company_name,
               d.sector,
               f.current_price_usd,
               f.price_change_pct,
               f.day_high_usd,
               f.day_low_usd,
               f.trading_volume,
               f.intraday_range_pct,
               f.price_momentum
        FROM   gold.fact_prices f
        JOIN   gold.dim_asset   d ON d.asset_id = f.asset_id
        WHERE  d.company_name IN ({placeholders})
        ORDER  BY f.price_change_pct DESC NULLS LAST
    """)
    return df.to_dict(orient="records")
