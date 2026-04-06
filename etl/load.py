"""
load.py — write the enriched master DataFrame into PostgreSQL.

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
from sqlalchemy import create_engine, text

_ENV_PATH = Path(__file__).parent.parent / ".env"

# ---------------------------------------------------------------------------
# Indicator metadata: column_name → (source, unit)
# Only columns listed here are written to the indicators table.
# ---------------------------------------------------------------------------

_INDICATOR_META: dict[str, tuple[str, str]] = {
    # World Bank WGI  (standardised score, −2.5 to +2.5)
    "control_of_corruption":          ("WGI",     "score"),
    "government_effectiveness":        ("WGI",     "score"),
    "political_stability":             ("WGI",     "score"),
    "regulatory_quality":              ("WGI",     "score"),
    "rule_of_law":                     ("WGI",     "score"),
    "voice_and_accountability":        ("WGI",     "score"),
    # IMF World Economic Outlook
    "gdp_growth_pct":                  ("IMF",     "percent"),
    "inflation_pct":                   ("IMF",     "percent"),
    "unemployment_pct":                ("IMF",     "percent"),
    "current_account_balance_usd_bn":  ("IMF",     "billion USD"),
    "gross_govt_debt_pct_gdp":         ("IMF",     "percent"),
    "gdp_usd_bn":                      ("IMF",     "billion USD"),
    # UNDP Human Development Index
    "hdi_value":                       ("UNDP",    "index"),
    "life_expectancy_years":           ("UNDP",    "years"),
    "expected_schooling_years":        ("UNDP",    "years"),
    "mean_schooling_years":            ("UNDP",    "years"),
    "gni_per_capita_2017ppp":          ("UNDP",    "2017 PPP USD"),
    # Polity5 political regime scores
    "polity2_score":                   ("Polity5", "score"),
    "democracy_score":                 ("Polity5", "score"),
    "autocracy_score":                 ("Polity5", "score"),
    # V-Dem democracy indices (0–1)
    "electoral_democracy_index":       ("V-Dem",   "index"),
    "liberal_democracy_index":         ("V-Dem",   "index"),
    "participatory_democracy_index":   ("V-Dem",   "index"),
    # Derived features (enrich.py)
    "gdp_growth_yoy_calc":             ("derived", "percent"),
    "governance_composite":            ("derived", "score"),
    "regional_avg_gdp_growth":         ("derived", "percent"),
    "regional_avg_governance":         ("derived", "score"),
}

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------


def _get_engine():
    """Build a SQLAlchemy engine from .env / environment variables."""
    load_dotenv(_ENV_PATH)
    host     = os.environ["DB_HOST"]
    port     = os.environ.get("DB_PORT", "5432")
    dbname   = os.environ["DB_NAME"]
    user     = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# Build tables as pandas DataFrames
# ---------------------------------------------------------------------------


def _build_countries(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy countries DataFrame with a sequential integer id.
    Rows where country_code is null or blank are excluded.
    continent column is optional — present only when enrich() has been run.
    """
    has_code = pdf["country_code"].notna() & (pdf["country_code"].str.strip() != "")

    cols = ["country_code", "country_name"]
    if "continent" in pdf.columns:
        cols.append("continent")

    countries = (
        pdf.loc[has_code, cols]
        .drop_duplicates(subset=["country_code"])
        .rename(columns={
            "country_code": "iso_code",
            "country_name": "name",
            "continent":    "region",
        })
        .reset_index(drop=True)
    )

    if "region" not in countries.columns:
        countries["region"] = None

    countries.insert(0, "id", countries.index + 1)
    return countries


def _build_indicators(pdf: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the wide DataFrame to long format, attach country_id via the
    countries table, and attach source/unit metadata.

    Rows with null values and rows whose country_code has no match in
    the countries table are dropped.
    """
    indicator_cols = [c for c in pdf.columns if c in _INDICATOR_META]

    long = pdf[["country_code", "year"] + indicator_cols].melt(
        id_vars=["country_code", "year"],
        value_vars=indicator_cols,
        var_name="indicator",
        value_name="value",
    )

    long = long.dropna(subset=["value", "country_code"])
    long = long[long["country_code"].str.strip() != ""]

    long["source"] = long["indicator"].map(lambda c: _INDICATOR_META[c][0])
    long["unit"]   = long["indicator"].map(lambda c: _INDICATOR_META[c][1])
    long["year"]   = long["year"].astype(int)

    # Join to get country_id; rows with no match are dropped
    id_map = countries.set_index("iso_code")["id"]
    long["country_id"] = long["country_code"].map(id_map)
    long = long.dropna(subset=["country_id"])
    long["country_id"] = long["country_id"].astype(int)

    return (
        long[["country_id", "indicator", "source", "year", "value", "unit"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load(enriched_df: DataFrame, chunksize: int = 2_000) -> None:
    """
    Write the enriched master DataFrame to PostgreSQL.

    Args:
        enriched_df : Spark DataFrame returned by enrich.enrich()
        chunksize   : rows per batch passed to pandas to_sql
    """
    print("=== Load ===")

    print("  [1/4] Collecting Spark DataFrame to pandas...")
    pdf = enriched_df.toPandas()
    print(f"        {len(pdf):,} rows  ×  {len(pdf.columns)} columns")

    print("  [2/4] Connecting to PostgreSQL...")
    engine = _get_engine()
    print(f"        connected  →  {engine.url.database}@{engine.url.host}")

    # Drop in FK-safe order so replace doesn't fail on dependent constraints
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS predictions CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS indicators  CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS countries   CASCADE"))

    print("  [3/4] Writing countries table...")
    countries = _build_countries(pdf)
    countries.to_sql("countries", engine, if_exists="replace", index=False, chunksize=chunksize)
    print(f"        {len(countries):,} rows written")

    print("  [4/4] Writing indicators table...")
    indicators = _build_indicators(pdf, countries)
    indicators.to_sql("indicators", engine, if_exists="replace", index=False, chunksize=chunksize)
    print(f"        {len(indicators):,} rows written")

    engine.dispose()
    print("\n  Done.")
