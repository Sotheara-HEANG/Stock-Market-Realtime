"""
test_source_client.py — unit tests for 01_ingestion/api_clients/source_client.py.

Verifies each client wraps the correct extractor and returns valid records.
No Kafka or network calls required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_ingestion"))

from api_clients.source_client import (
    SOURCE_CLIENTS,
    WGIClient,
    IMFClient,
    HDIClient,
    Polity5Client,
    VDemClient,
    get_client,
)

EXPECTED_KEYS = {"country_code", "country_name", "indicator", "year", "value", "source"}


def _assert_records(records: list[dict], name: str) -> None:
    assert len(records) > 0, f"{name}: returned no records"
    for r in records[:5]:   # spot-check first 5
        assert set(r.keys()) >= EXPECTED_KEYS, f"{name}: missing keys in {r}"
        assert r["value"] is not None, f"{name}: null value in {r}"
        assert r["year"]  is not None, f"{name}: null year in {r}"


# ---------------------------------------------------------------------------
# Individual clients
# ---------------------------------------------------------------------------

def test_wgi_client_fetch():
    records = WGIClient().fetch()
    _assert_records(records, "WGI")


def test_imf_client_fetch():
    records = IMFClient().fetch()
    _assert_records(records, "IMF")


def test_hdi_client_fetch():
    records = HDIClient().fetch()
    _assert_records(records, "HDI")


def test_polity5_client_fetch():
    records = Polity5Client().fetch()
    _assert_records(records, "Polity5")


def test_vdem_client_fetch():
    records = VDemClient().fetch()
    _assert_records(records, "V-Dem")


# ---------------------------------------------------------------------------
# get_client factory
# ---------------------------------------------------------------------------

def test_get_client_returns_correct_types():
    assert isinstance(get_client("wgi"),    WGIClient)
    assert isinstance(get_client("imf"),    IMFClient)
    assert isinstance(get_client("hdi"),    HDIClient)
    assert isinstance(get_client("polity5"), Polity5Client)
    assert isinstance(get_client("vdem"),   VDemClient)


def test_get_client_unknown_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        get_client("nonexistent_source")


def test_source_clients_registry_complete():
    assert set(SOURCE_CLIENTS.keys()) == {"wgi", "imf", "hdi", "polity5", "vdem", "wb_api"}
