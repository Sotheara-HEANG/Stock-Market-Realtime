"""
raw.py — load each source file as-is into a pandas DataFrame, then save to data/raw/.

No cleaning, no column renaming, no melting, no dropna.
Preserves the original structure of every source exactly as received.

Output: data/raw/<source>.parquet  (one file per source)

Usage:
    from etl.raw import load_raw, save_raw_snapshots

    frames = load_raw()          # dict of raw DataFrames (from disk if cached)
    save_raw_snapshots()         # fetch/read all sources and write parquet files
    save_raw_snapshots(api=True) # also hit the World Bank live API
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATASET_DIR = Path(__file__).resolve().parents[1] / "Dataset"
RAW_DIR     = Path(__file__).resolve().parents[1] / "data" / "raw"

# Shared session — same SSL bypass as extract.py / Unemployment_Economic_Indicators project
_session = requests.Session()
_session.verify = False


# ---------------------------------------------------------------------------
# WGI — one raw DataFrame per CSV file (wide format, metadata header stripped)
# ---------------------------------------------------------------------------

_WGI_FILES = {
    "wgi_control_of_corruption":    "WGI_Control_of_Corruption.csv",
    "wgi_government_effectiveness": "WGI_Government_Effectiveness.csv",
    "wgi_political_stability":      "WGI_Political_Stability.csv",
    "wgi_regulatory_quality":       "WGI_Regulatory_Quality.csv",
    "wgi_rule_of_law":              "WGI_Rule_of_Law.csv",
    "wgi_voice_accountability":     "WGI_Voice_Accountability.csv",
}


def _load_wgi_raw() -> dict[str, pd.DataFrame]:
    """Return one DataFrame per WGI file. Skips the 4-row metadata header only."""
    frames = {}
    for key, filename in _WGI_FILES.items():
        path = DATASET_DIR / filename
        # skiprows=4 removes the World Bank boilerplate; everything else untouched
        frames[key] = pd.read_csv(path, skiprows=4, encoding="utf-8-sig")
    return frames


# ---------------------------------------------------------------------------
# IMF WEO — already a clean wide CSV, load as-is
# ---------------------------------------------------------------------------

def _load_imf_raw() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "IMF_WEO_2024.csv")


# ---------------------------------------------------------------------------
# UNDP HDI — compound multi-row header; load with header=None to see all rows
# ---------------------------------------------------------------------------

def _load_hdi_raw() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "UNDP_HDI_2023-24.csv", header=None)


# ---------------------------------------------------------------------------
# Polity5 — long format with all 37 original columns
# ---------------------------------------------------------------------------

def _load_polity5_raw() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "Polity5.csv", low_memory=False)


# ---------------------------------------------------------------------------
# V-Dem — long format, 4 columns (Entity, Code, Year, 3 indices)
# ---------------------------------------------------------------------------

def _load_vdem_raw() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "VDem_Core_Indices.csv")


# ---------------------------------------------------------------------------
# Freedom House FIW — included in Dataset, load as-is
# ---------------------------------------------------------------------------

def _load_freedom_house_raw() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "Freedom_House_FIW_2013-2025.csv")


# ---------------------------------------------------------------------------
# World Bank API — raw JSON response fields, all countries (no ISO filter)
# ---------------------------------------------------------------------------

_WB_API_INDICATORS = {
    "CC.EST": "control_of_corruption",
    "GE.EST": "government_effectiveness",
    "PV.EST": "political_stability",
    "RQ.EST": "regulatory_quality",
    "RL.EST": "rule_of_law",
    "VA.EST": "voice_and_accountability",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "NY.GDP.PCAP.CD":     "gdp_per_capita_usd",
    "FP.CPI.TOTL.ZG":    "inflation_cpi_pct",
    "SL.UEM.TOTL.ZS":    "unemployment_pct",
    "GC.DOD.TOTL.GD.ZS": "public_debt_pct_gdp",
    "NE.TRD.GNFS.ZS":    "trade_pct_gdp",
}

_WB_API_BASE   = "https://api.worldbank.org/v2/country/all/indicator"
_WB_API_PARAMS = "format=json&per_page=20000&mrv=30"


def _load_wb_api_raw() -> pd.DataFrame:
    """
    Fetch raw JSON from World Bank API and flatten into a DataFrame.
    All original fields kept (countryiso3code, country.id, country.value,
    indicator.id, indicator.value, date, value, unit, obs_status, decimal).
    """
    frames: list[pd.DataFrame] = []

    for wb_code, label in _WB_API_INDICATORS.items():
        url = f"{_WB_API_BASE}/{wb_code}?{_WB_API_PARAMS}"
        try:
            resp = _session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if len(data) < 2 or not data[1]:
                print(f"  [skip] {label}: no data")
                continue

            rows = []
            for d in data[1]:
                rows.append({
                    "indicator_id":    d["indicator"]["id"],
                    "indicator_label": label,
                    "country_id":      d["country"]["id"],
                    "country_name":    d["country"]["value"],
                    "country_iso3":    d.get("countryiso3code", ""),
                    "date":            d.get("date"),
                    "value":           d.get("value"),
                    "unit":            d.get("unit", ""),
                    "obs_status":      d.get("obs_status", ""),
                    "decimal":         d.get("decimal"),
                })

            df = pd.DataFrame(rows)
            frames.append(df)
            print(f"  [ok] {label}: {len(df):,} rows")

        except requests.RequestException as exc:
            print(f"  [error] {label}: {exc}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_raw_snapshots(api: bool = False) -> None:
    """
    Load every source and write to data/raw/<name>.parquet.

    Args:
        api: if True, also fetches live data from the World Bank API
             and saves as data/raw/wb_api.parquet.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # WGI (6 files → 6 parquet files)
    print("Loading WGI...")
    for key, df in _load_wgi_raw().items():
        out = RAW_DIR / f"{key}.parquet"
        df.to_parquet(out, index=False)
        print(f"  saved {out.name}  {df.shape}")

    # IMF
    print("Loading IMF WEO...")
    df = _load_imf_raw()
    out = RAW_DIR / "imf_weo.parquet"
    df.to_parquet(out, index=False)
    print(f"  saved {out.name}  {df.shape}")

    # HDI
    print("Loading UNDP HDI...")
    df = _load_hdi_raw()
    out = RAW_DIR / "hdi.parquet"
    df.to_parquet(out, index=False)
    print(f"  saved {out.name}  {df.shape}")

    # Polity5
    print("Loading Polity5...")
    df = _load_polity5_raw()
    out = RAW_DIR / "polity5.parquet"
    df.to_parquet(out, index=False)
    print(f"  saved {out.name}  {df.shape}")

    # V-Dem
    print("Loading V-Dem...")
    df = _load_vdem_raw()
    out = RAW_DIR / "vdem.parquet"
    df.to_parquet(out, index=False)
    print(f"  saved {out.name}  {df.shape}")

    # Freedom House
    print("Loading Freedom House FIW...")
    df = _load_freedom_house_raw()
    out = RAW_DIR / "freedom_house.parquet"
    df.to_parquet(out, index=False)
    print(f"  saved {out.name}  {df.shape}")

    # World Bank API (optional — live network call)
    if api:
        print("Fetching World Bank API...")
        df = _load_wb_api_raw()
        if not df.empty:
            out = RAW_DIR / "wb_api.parquet"
            df.to_parquet(out, index=False)
            print(f"  saved {out.name}  {df.shape}")

    print(f"\nDone. Raw snapshots in: {RAW_DIR}")


def load_raw(api: bool = False) -> dict[str, pd.DataFrame]:
    """
    Load raw DataFrames from data/raw/*.parquet (runs save_raw_snapshots first
    if the directory is empty).

    Returns a dict keyed by source name.
    """
    parquet_files = list(RAW_DIR.glob("*.parquet"))

    if not parquet_files:
        print("data/raw/ is empty — running save_raw_snapshots()...")
        save_raw_snapshots(api=api)
        parquet_files = list(RAW_DIR.glob("*.parquet"))

    return {p.stem: pd.read_parquet(p) for p in sorted(parquet_files)}
