-- Bronze layer: raw Finnhub stock metrics.
-- One row per (ticker symbol, indicator, year) metric.

CREATE SCHEMA IF NOT EXISTS bronze;

DROP TABLE IF EXISTS bronze.raw_commodity_prices CASCADE;

CREATE TABLE bronze.raw_commodity_prices (
    id             SERIAL       PRIMARY KEY,
    symbol         VARCHAR(40)  NOT NULL,
    commodity_name VARCHAR(160),
    indicator      VARCHAR(100) NOT NULL,
    year           INT          NOT NULL,
    value          FLOAT,
    source         VARCHAR(100),
    ingested_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_bronze_commodity_symbol    ON bronze.raw_commodity_prices (symbol);
CREATE INDEX idx_bronze_commodity_indicator ON bronze.raw_commodity_prices (indicator);
CREATE INDEX idx_bronze_commodity_ingested  ON bronze.raw_commodity_prices (ingested_at);
