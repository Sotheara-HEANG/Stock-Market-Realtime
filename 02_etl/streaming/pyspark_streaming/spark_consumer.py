"""
spark_consumer.py — PySpark Structured Streaming: Kafka raw topics → Bronze table.

Reads all raw.* Kafka topics, validates the 6-field schema, and writes
micro-batches to the PostgreSQL bronze.raw_indicators table.

Usage:
    spark-submit \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,\
org.postgresql:postgresql:42.6.0 \
        02_etl/streaming/pyspark_streaming/spark_consumer.py

    # Run in test mode (console sink, no DB writes)
    python 02_etl/streaming/pyspark_streaming/spark_consumer.py --test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_ROOT / ".env")

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DB_HOST    = os.environ.get("DB_HOST", "localhost")
DB_PORT    = os.environ.get("DB_PORT", "5432")
DB_NAME    = os.environ.get("DB_NAME", "econ_pipeline")
DB_USER    = os.environ.get("DB_USER", "kongsattha")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_URL     = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"

# All raw source topics
RAW_TOPICS = "raw.wgi,raw.imf,raw.hdi,raw.polity5,raw.vdem,raw.wb_api"

# Expected JSON schema for each record published by the producer
_RECORD_SCHEMA = StructType([
    StructField("country_code", StringType(),  True),
    StructField("country_name", StringType(),  True),
    StructField("indicator",    StringType(),  True),
    StructField("year",         IntegerType(), True),
    StructField("value",        DoubleType(),  True),
    StructField("source",       StringType(),  True),
])


def _get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("econ-governance-streaming")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def _read_kafka(spark: SparkSession):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", RAW_TOPICS)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )


def _parse_records(raw_df):
    """Deserialise JSON value column into typed fields."""
    parsed = raw_df.select(
        F.col("topic"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.from_json(F.col("value").cast("string"), _RECORD_SCHEMA).alias("data"),
    ).select(
        "topic",
        "kafka_timestamp",
        "data.*",
    )

    # Basic validation — drop rows missing mandatory fields
    return parsed.filter(
        F.col("country_code").isNotNull()
        & F.col("indicator").isNotNull()
        & F.col("year").isNotNull()
        & F.col("value").isNotNull()
    ).withColumn("ingested_at", F.current_timestamp())


def _write_to_bronze(batch_df, batch_id: int) -> None:
    """Write each micro-batch to the bronze.raw_indicators table."""
    if batch_df.isEmpty():
        return
    (
        batch_df
        .select(
            "country_code", "country_name", "indicator",
            "year", "value", "source", "topic", "kafka_timestamp", "ingested_at",
        )
        .write
        .format("jdbc")
        .option("url", DB_URL)
        .option("dbtable", "bronze.raw_indicators")
        .option("user", DB_USER)
        .option("password", DB_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .mode("append")
        .save()
    )
    print(f"  Batch {batch_id}: wrote {batch_df.count():,} rows to bronze.raw_indicators")


def run(test_mode: bool = False) -> None:
    spark = _get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading from Kafka topics: {RAW_TOPICS}")
    raw = _read_kafka(spark)
    parsed = _parse_records(raw)

    if test_mode:
        query = (
            parsed.writeStream
            .outputMode("append")
            .format("console")
            .option("truncate", False)
            .trigger(once=True)
            .start()
        )
    else:
        query = (
            parsed.writeStream
            .outputMode("append")
            .foreachBatch(_write_to_bronze)
            .trigger(processingTime="30 seconds")
            .option("checkpointLocation", str(_ROOT / "data" / "checkpoints" / "streaming"))
            .start()
        )

    query.awaitTermination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Print to console instead of writing to DB")
    args = parser.parse_args()
    run(test_mode=args.test)
