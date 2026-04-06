"""
test_extract.py — verify each extractor produces the correct 6-column schema.

Tests use the real Dataset/ files committed to the repository.
No mocking: if the source CSV changes structure, the test catches it.
"""

import pandas as pd
import pytest

from etl.extract import (
    extract_all,
    extract_hdi,
    extract_imf,
    extract_polity5,
    extract_vdem,
    extract_wgi,
)

EXPECTED_COLS = ["country_code", "country_name", "indicator", "year", "value", "source"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_schema(df: pd.DataFrame, name: str) -> None:
    assert list(df.columns) == EXPECTED_COLS, f"{name}: wrong columns {list(df.columns)}"
    assert len(df) > 0, f"{name}: returned empty DataFrame"
    assert df["value"].notna().all(), f"{name}: contains null values after extraction"
    assert df["year"].notna().all(), f"{name}: contains null years"


# ---------------------------------------------------------------------------
# Per-source tests
# ---------------------------------------------------------------------------

def test_extract_wgi_schema():
    df = extract_wgi()
    assert_schema(df, "WGI")
    # WGI has exactly 6 governance indicators
    assert df["indicator"].nunique() == 6


def test_extract_wgi_country_codes_are_iso3():
    df = extract_wgi()
    bad = df[~df["country_code"].str.match(r"^[A-Z]{3}$")]
    assert len(bad) == 0, f"Non-ISO-3 codes found: {bad['country_code'].unique()}"


def test_extract_imf_schema():
    df = extract_imf()
    assert_schema(df, "IMF")
    assert df["source"].eq("IMF WEO 2024").all()


def test_extract_hdi_schema():
    df = extract_hdi()
    assert_schema(df, "HDI")
    assert df["source"].eq("UNDP HDI 2023-24").all()
    # All HDI rows are for a single year
    assert df["year"].nunique() == 1


def test_extract_polity5_schema():
    df = extract_polity5()
    assert_schema(df, "Polity5")
    assert df["source"].eq("Polity5").all()
    # Polity5 covers a wide historical range
    assert df["year"].min() < 1900


def test_extract_vdem_schema():
    df = extract_vdem()
    assert_schema(df, "V-Dem")
    assert df["source"].eq("V-Dem").all()
    assert df["indicator"].nunique() == 3


# ---------------------------------------------------------------------------
# Combined extract_all test
# ---------------------------------------------------------------------------

def test_extract_all_returns_all_sources():
    result = extract_all(include_api=False)
    assert set(result.keys()) == {"wgi", "imf", "hdi", "polity5", "vdem"}


def test_extract_all_schemas():
    result = extract_all(include_api=False)
    for name, df in result.items():
        assert_schema(df, name)


def test_extract_all_total_rows():
    result = extract_all(include_api=False)
    total = sum(len(df) for df in result.values())
    assert total > 50_000, f"Expected >50k rows total, got {total:,}"
