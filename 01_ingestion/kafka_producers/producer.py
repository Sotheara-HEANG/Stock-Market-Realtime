"""
producer.py — Kafka producers for all economic/governance data sources.

Reads source data via the API clients, serialises each record as JSON,
and publishes to the corresponding Kafka topic defined in sources.yaml.

Usage:
    # Publish all sources
    python 01_ingestion/kafka_producers/producer.py

    # Publish a specific source
    python 01_ingestion/kafka_producers/producer.py --source wgi

    # Include live World Bank API call
    python 01_ingestion/kafka_producers/producer.py --source wb_api
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_clients.source_client import SOURCE_CLIENTS, get_client

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
    )


def publish_source(producer: KafkaProducer, source: str, topic: str) -> int:
    """Fetch all records for a source and publish them to the Kafka topic."""
    print(f"  [{source}] Fetching data...")
    client = get_client(source)
    records = client.fetch()
    print(f"  [{source}] Publishing {len(records):,} records → topic '{topic}'")

    for record in records:
        key = record.get("country_code") or "unknown"
        producer.send(topic, key=key, value=record)

    producer.flush()
    print(f"  [{source}] Done.")
    return len(records)


def run(sources: list[str] | None = None) -> None:
    config = _load_config()
    bootstrap = config["kafka"]["bootstrap_servers"]
    topics: dict[str, str] = config["kafka"]["topics"]

    if sources is None:
        sources = list(SOURCE_CLIENTS.keys())

    print(f"Connecting to Kafka at {bootstrap}...")
    try:
        producer = _make_producer(bootstrap)
    except NoBrokersAvailable:
        print(f"ERROR: Cannot reach Kafka broker at {bootstrap}.")
        print("Start Kafka first:  docker compose up -d kafka")
        sys.exit(1)

    total = 0
    start = time.time()
    for source in sources:
        if source not in topics:
            print(f"  [skip] No topic configured for '{source}'")
            continue
        total += publish_source(producer, source, topics[source])

    elapsed = time.time() - start
    print(f"\nPublished {total:,} records in {elapsed:.1f}s")
    producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish economic/governance data to Kafka")
    parser.add_argument(
        "--source",
        choices=list(SOURCE_CLIENTS.keys()),
        help="Publish a single source only. Defaults to all sources.",
    )
    args = parser.parse_args()
    run(sources=[args.source] if args.source else None)
