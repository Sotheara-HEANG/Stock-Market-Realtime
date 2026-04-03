"""
extract.py — load raw source files into pandas DataFrames.

Each function returns a tidy long-format DataFrame with consistent columns:
    country_code : ISO 3-letter code (or Polity scode where ISO unavailable)
    country_name : human-readable name
    indicator    : snake_case indicator label
    year         : int
    value        : float
    source       : source label string

Usage:
    from etl.extract import extract_wgi, extract_imf, extract_hdi, extract_polity5
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[1] / "Dataset"

# ---------------------------------------------------------------------------
# WGI — World Bank World Governance Indicators (6 CSV files, wide format)
# ---------------------------------------------------------------------------

_WGI_FILES = {
    "control_of_corruption":      "WGI_Control_of_Corruption.csv",
    "government_effectiveness":   "WGI_Government_Effectiveness.csv",
    "political_stability":        "WGI_Political_Stability.csv",
    "regulatory_quality":         "WGI_Regulatory_Quality.csv",
    "rule_of_law":                "WGI_Rule_of_Law.csv",
    "voice_and_accountability":   "WGI_Voice_Accountability.csv",
}

# Year columns present in WGI files
_WGI_YEAR_RANGE = [str(y) for y in range(1960, 2026)]


def extract_wgi() -> pd.DataFrame:
    """
    Read all 6 WGI CSVs and return a single long-format DataFrame.

    Source format: 4-row metadata header, then wide with one year per column.
    """
    frames: list[pd.DataFrame] = []

    for indicator, filename in _WGI_FILES.items():
        path = DATASET_DIR / filename
        raw = pd.read_csv(path, skiprows=4, encoding="utf-8-sig")

        # Keep only country rows (Country Code is a 3-letter ISO code)
        raw = raw.dropna(subset=["Country Code"])
        raw = raw[raw["Country Code"].str.match(r"^[A-Z]{3}$", na=False)]

        # Select only year columns that actually exist in this file
        year_cols = [c for c in _WGI_YEAR_RANGE if c in raw.columns]

        melted = raw.melt(
            id_vars=["Country Name", "Country Code"],
            value_vars=year_cols,
            var_name="year",
            value_name="value",
        )
        melted["indicator"] = indicator
        frames.append(melted)

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Country Name": "country_name", "Country Code": "country_code"})
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["source"] = "World Bank WGI"
    df = df.dropna(subset=["value"])
    return df[["country_code", "country_name", "indicator", "year", "value", "source"]]


# ---------------------------------------------------------------------------
# IMF — World Economic Outlook 2024 (already long format)
# ---------------------------------------------------------------------------

_IMF_INDICATORS = {
    "GDP_Growth_Rate_pct":             "gdp_growth_pct",
    "Inflation_Rate_pct":              "inflation_pct",
    "Unemployment_Rate_pct":           "unemployment_pct",
    "Current_Account_Balance_USD_bn":  "current_account_balance_usd_bn",
    "Gross_Govt_Debt_pct_GDP":         "gross_govt_debt_pct_gdp",
    "GDP_Current_Prices_USD_bn":       "gdp_usd_bn",
}


def extract_imf() -> pd.DataFrame:
    """
    Read IMF_WEO_2024.csv and return long-format DataFrame.

    Source format: already long, one row per (country_code, year), multiple
    indicator columns — melt into one row per (country_code, year, indicator).
    """
    path = DATASET_DIR / "IMF_WEO_2024.csv"
    raw = pd.read_csv(path)

    raw = raw.rename(columns={"country": "country_code"})
    raw["country_code"] = raw["country_code"].str.upper().str.strip()

    value_cols = list(_IMF_INDICATORS.keys())
    df = raw.melt(
        id_vars=["country_code", "year"],
        value_vars=value_cols,
        var_name="indicator",
        value_name="value",
    )
    df["indicator"] = df["indicator"].map(_IMF_INDICATORS)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["source"] = "IMF WEO 2024"
    df["country_name"] = ""   # IMF file has no name column; join via countries table later
    df = df.dropna(subset=["value"])
    return df[["country_code", "country_name", "indicator", "year", "value", "source"]]


# ---------------------------------------------------------------------------
# UNDP HDI — Human Development Index 2023-24 (complex multi-row header)
# ---------------------------------------------------------------------------

# Column positions in the raw file (0-indexed) after skipping the first 4 rows
_HDI_COL_MAP = {
    1:  "country_name",
    2:  "hdi_value",
    4:  "life_expectancy_years",
    6:  "expected_schooling_years",
    8:  "mean_schooling_years",
    10: "gni_per_capita_2017ppp",
}
_HDI_YEAR = 2022   # all values in this edition are for 2022


def extract_hdi() -> pd.DataFrame:
    """
    Read UNDP_HDI_2023-24.csv and return long-format DataFrame.

    Source format: 4-row compound header (indicator name / sub-label / year / category row),
    then data rows. Column positions are used directly because merged headers lack clean names.
    """
    path = DATASET_DIR / "UNDP_HDI_2023-24.csv"
    raw = pd.read_csv(path, header=None, skiprows=4, dtype=str)

    # Select only the columns we care about and rename
    cols_needed = list(_HDI_COL_MAP.keys())
    df = raw.iloc[:, cols_needed].copy()
    df.columns = list(_HDI_COL_MAP.values())

    # Drop section header rows (e.g. "VERY HIGH HUMAN DEVELOPMENT") — they have no HDI value
    df = df[pd.to_numeric(df["hdi_value"], errors="coerce").notna()].copy()
    df["country_name"] = df["country_name"].str.strip()
    df = df[df["country_name"].notna() & (df["country_name"] != "")]

    # Melt indicator columns into long format
    indicator_cols = [c for c in df.columns if c != "country_name"]
    df_long = df.melt(
        id_vars=["country_name"],
        value_vars=indicator_cols,
        var_name="indicator",
        value_name="value",
    )
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long["year"] = _HDI_YEAR
    df_long["country_code"] = ""   # HDI file has no ISO code; fill via fuzzy join if needed
    df_long["source"] = "UNDP HDI 2023-24"
    df_long = df_long.dropna(subset=["value"])
    return df_long[["country_code", "country_name", "indicator", "year", "value", "source"]]


# ---------------------------------------------------------------------------
# Polity5 — Political regime scores (long format, many columns)
# ---------------------------------------------------------------------------

_POLITY5_COLS = {
    "country": "country_name",
    "scode":   "country_code",   # Polity alpha code (not ISO-3; closest available)
    "year":    "year",
    "polity2": "polity2_score",  # Main democracy scale: -10 (autocracy) to +10 (democracy)
    "democ":   "democracy_score",
    "autoc":   "autocracy_score",
}

# Polity5 uses -66, -77, -88 as special missing codes
_POLITY5_MISSING = {-66, -77, -88}


def extract_polity5() -> pd.DataFrame:
    """
    Read Polity5.csv and return long-format DataFrame.

    Source format: already long, one row per (country, year). Key columns:
    polity2 (-10 to +10), democ, autoc. Special values -66/-77/-88 = missing.
    Note: scode is the Polity alpha code, not ISO-3.
    """
    path = DATASET_DIR / "Polity5.csv"
    raw = pd.read_csv(path, low_memory=False)

    df = raw[list(_POLITY5_COLS.keys())].copy()
    df = df.rename(columns=_POLITY5_COLS)

    indicator_cols = ["polity2_score", "democracy_score", "autocracy_score"]
    for col in indicator_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(~df[col].isin(_POLITY5_MISSING))

    df_long = df.melt(
        id_vars=["country_code", "country_name", "year"],
        value_vars=indicator_cols,
        var_name="indicator",
        value_name="value",
    )
    df_long["source"] = "Polity5"
    df_long = df_long.dropna(subset=["value"])
    return df_long[["country_code", "country_name", "indicator", "year", "value", "source"]]


# ---------------------------------------------------------------------------
# Convenience: load everything at once
# ---------------------------------------------------------------------------

def extract_all() -> dict[str, pd.DataFrame]:
    """Return a dict of all source DataFrames keyed by source name."""
    return {
        "wgi":     extract_wgi(),
        "imf":     extract_imf(),
        "hdi":     extract_hdi(),
        "polity5": extract_polity5(),
    }
