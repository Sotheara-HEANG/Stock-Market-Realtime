"""
source_client.py — thin wrappers around the ETL extractors.

Each client exposes a fetch() method that returns a list of JSON-serialisable
records in the shared 6-field schema:
    {country_code, country_name, indicator, year, value, source}

These records are what the Kafka producers publish.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from etl.extract import (
    extract_wgi,
    extract_imf,
    extract_hdi,
    extract_polity5,
    extract_vdem,
    extract_wb_api,
)


def _to_records(df) -> list[dict]:
    """Convert a pandas DataFrame to a list of plain dicts."""
    return df.to_dict(orient="records")


class WGIClient:
    """World Bank World Governance Indicators — reads local CSV files."""

    def fetch(self) -> list[dict]:
        return _to_records(extract_wgi())


class IMFClient:
    """IMF World Economic Outlook — reads local CSV file."""

    def fetch(self) -> list[dict]:
        return _to_records(extract_imf())


class HDIClient:
    """UNDP Human Development Index — reads local CSV file."""

    def fetch(self) -> list[dict]:
        return _to_records(extract_hdi())


class Polity5Client:
    """Polity5 political regime scores — reads local CSV file."""

    def fetch(self) -> list[dict]:
        return _to_records(extract_polity5())


class VDemClient:
    """V-Dem democracy indices — reads local CSV file."""

    def fetch(self) -> list[dict]:
        return _to_records(extract_vdem())


class WorldBankAPIClient:
    """World Bank WDI REST API — live HTTP requests."""

    def __init__(self, indicators: dict[str, str] | None = None):
        self._indicators = indicators

    def fetch(self) -> list[dict]:
        return _to_records(extract_wb_api(self._indicators))


# Map source name → client class (matches sources.yaml keys)
SOURCE_CLIENTS: dict[str, type] = {
    "wgi":    WGIClient,
    "imf":    IMFClient,
    "hdi":    HDIClient,
    "polity5": Polity5Client,
    "vdem":   VDemClient,
    "wb_api": WorldBankAPIClient,
}


def get_client(source: str):
    """Return an instantiated client for the given source name."""
    cls = SOURCE_CLIENTS.get(source)
    if cls is None:
        raise ValueError(f"Unknown source '{source}'. Choose from: {list(SOURCE_CLIENTS)}")
    return cls()
