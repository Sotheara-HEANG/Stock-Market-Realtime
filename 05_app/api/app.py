"""
app.py — FastAPI backend for the economic/governance pipeline.

Endpoints:
  GET /health
  GET /countries                         — list all countries
  GET /indicators                        — list available indicator names
  GET /data/{iso_code}                   — all indicators for a country
  GET /data/{iso_code}/{indicator}       — time-series for one country + indicator
  GET /predictions/{iso_code}/{indicator}— forecasts for one country + indicator
  GET /compare?countries=USA,CHN&indicator=gdp_growth_pct
  GET /top?indicator=governance_composite&year=2022&n=10

Usage:
    uvicorn 05_app.api.app:app --reload --port 8000
    # or from project root:
    python -m uvicorn 05_app.api.app:app --reload
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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

DB_HOST     = os.environ.get("DB_HOST", "localhost")
DB_PORT     = os.environ.get("DB_PORT", "5432")
DB_NAME     = os.environ.get("DB_NAME", "econ_pipeline")
DB_USER     = os.environ.get("DB_USER", "kongsattha")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_URL      = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

app = FastAPI(
    title="Economic & Governance Pipeline API",
    description="Query indicators and forecasts from the econ-governance data pipeline.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True)
    return _engine


def _query(sql: str, params: dict | None = None) -> list[dict]:
    with get_engine().connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        cols = list(result.keys())
    return [dict(zip(cols, row)) for row in rows]


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        _query("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        raise HTTPException(503, detail=f"DB connection failed: {e}")


# ── Countries ────────────────────────────────────────────────────────────────

@app.get("/countries")
def list_countries():
    """List all countries in the dataset."""
    return _query(
        "SELECT id, iso_code, name, region FROM countries ORDER BY name"
    )


# ── Indicators ───────────────────────────────────────────────────────────────

@app.get("/indicators")
def list_indicators():
    """List all distinct indicator names and their sources."""
    return _query(
        "SELECT DISTINCT indicator, source FROM indicators ORDER BY source, indicator"
    )


# ── Time-series data ──────────────────────────────────────────────────────────

@app.get("/data/{iso_code}")
def get_country_data(iso_code: str):
    """All indicator values for a single country, ordered by indicator + year."""
    rows = _query(
        """
        SELECT i.indicator, i.source, i.year, i.value, i.unit
        FROM   indicators i
        JOIN   countries  c ON c.id = i.country_id
        WHERE  UPPER(c.iso_code) = UPPER(:iso)
        ORDER  BY i.indicator, i.year
        """,
        {"iso": iso_code},
    )
    if not rows:
        raise HTTPException(404, detail=f"No data found for country '{iso_code}'")
    return {"iso_code": iso_code.upper(), "records": rows}


@app.get("/data/{iso_code}/{indicator}")
def get_indicator_series(iso_code: str, indicator: str):
    """Time-series for one country + indicator."""
    rows = _query(
        """
        SELECT i.year, i.value, i.source, i.unit
        FROM   indicators i
        JOIN   countries  c ON c.id = i.country_id
        WHERE  UPPER(c.iso_code) = UPPER(:iso)
          AND  i.indicator = :ind
        ORDER  BY i.year
        """,
        {"iso": iso_code, "ind": indicator},
    )
    if not rows:
        raise HTTPException(
            404, detail=f"No data for country='{iso_code}' indicator='{indicator}'"
        )
    return {"iso_code": iso_code.upper(), "indicator": indicator, "series": rows}


# ── Predictions ───────────────────────────────────────────────────────────────

@app.get("/predictions/{iso_code}/{indicator}")
def get_predictions(iso_code: str, indicator: str):
    """Forecast values for one country + indicator (all models)."""
    rows = _query(
        """
        SELECT p.model_name, p.predicted_year, p.predicted_value,
               p.confidence_low, p.confidence_high, p.run_at
        FROM   predictions p
        JOIN   countries   c ON c.id = p.country_id
        WHERE  UPPER(c.iso_code) = UPPER(:iso)
          AND  p.indicator = :ind
        ORDER  BY p.model_name, p.predicted_year
        """,
        {"iso": iso_code, "ind": indicator},
    )
    if not rows:
        raise HTTPException(
            404, detail=f"No predictions for country='{iso_code}' indicator='{indicator}'"
        )
    return {"iso_code": iso_code.upper(), "indicator": indicator, "forecasts": rows}


# ── Compare ───────────────────────────────────────────────────────────────────

@app.get("/compare")
def compare_countries(
    countries: str = Query(..., description="Comma-separated ISO codes, e.g. USA,CHN,DEU"),
    indicator: str = Query(..., description="Indicator name"),
):
    """Compare one indicator across multiple countries."""
    iso_list = [c.strip().upper() for c in countries.split(",")]
    placeholders = ", ".join(f"'{c}'" for c in iso_list)
    rows = _query(
        f"""
        SELECT c.iso_code, c.name, i.year, i.value
        FROM   indicators i
        JOIN   countries  c ON c.id = i.country_id
        WHERE  UPPER(c.iso_code) IN ({placeholders})
          AND  i.indicator = :ind
        ORDER  BY c.iso_code, i.year
        """,
        {"ind": indicator},
    )
    return {"indicator": indicator, "countries": iso_list, "series": rows}


# ── Top N ─────────────────────────────────────────────────────────────────────

@app.get("/top")
def top_countries(
    indicator: str = Query(..., description="Indicator name"),
    year: int      = Query(..., description="Year"),
    n: int         = Query(10,  description="Number of results", ge=1, le=50),
    ascending: bool = Query(False, description="Sort ascending (lowest first)"),
):
    """Top N countries for an indicator in a given year."""
    order = "ASC" if ascending else "DESC"
    rows = _query(
        f"""
        SELECT c.iso_code, c.name, c.region, i.value
        FROM   indicators i
        JOIN   countries  c ON c.id = i.country_id
        WHERE  i.indicator = :ind
          AND  i.year = :yr
        ORDER  BY i.value {order} NULLS LAST
        LIMIT  :n
        """,
        {"ind": indicator, "yr": year, "n": n},
    )
    return {"indicator": indicator, "year": year, "top": rows}
