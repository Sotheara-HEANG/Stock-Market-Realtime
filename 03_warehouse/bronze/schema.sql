-- Bronze layer: raw Finnhub stock metrics.
-- One row per (ticker symbol, indicator, timeframe, time_index) metric.

CREATE SCHEMA IF NOT EXISTS bronze;

DROP TABLE IF EXISTS bronze.raw_commodity_prices CASCADE;

CREATE TABLE bronze.raw_commodity_prices (
    id             SERIAL       PRIMARY KEY,
    symbol         VARCHAR(40)  NOT NULL,
    commodity_name VARCHAR(160),
    indicator      VARCHAR(100) NOT NULL,
    timeframe      VARCHAR(10)  NOT NULL,
    time_index     DATE         NOT NULL,
    value          FLOAT,
    source         VARCHAR(100),
    ingested_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_bronze_commodity_symbol    ON bronze.raw_commodity_prices (symbol);
CREATE INDEX idx_bronze_commodity_tf_index  ON bronze.raw_commodity_prices (timeframe, time_index);
CREATE INDEX idx_bronze_commodity_ingested  ON bronze.raw_commodity_prices (ingested_at);
