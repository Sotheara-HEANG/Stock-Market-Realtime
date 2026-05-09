-- ============================================================
-- Bronze Layer — Raw / Staging
-- Stores records exactly as received from Kafka topics.
-- No transformation applied. Append-only.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS bronze;

-- Raw indicator records ingested from all Kafka topics
CREATE TABLE IF NOT EXISTS bronze.raw_indicators (
    id              BIGSERIAL    PRIMARY KEY,
    country_code    VARCHAR(10),
    country_name    VARCHAR(150),
    indicator       VARCHAR(100),
    year            SMALLINT,
    value           DOUBLE PRECISION,
    source          VARCHAR(100),
    topic           VARCHAR(100),           -- Kafka topic the record came from
    kafka_timestamp TIMESTAMPTZ,            -- Kafka event timestamp
    ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Partition hint for large tables (optional — apply after initial load)
-- CREATE INDEX IF NOT EXISTS idx_bronze_source    ON bronze.raw_indicators(source);
-- CREATE INDEX IF NOT EXISTS idx_bronze_year      ON bronze.raw_indicators(year);
-- CREATE INDEX IF NOT EXISTS idx_bronze_indicator ON bronze.raw_indicators(indicator);

-- Batch file load tracking
CREATE TABLE IF NOT EXISTS bronze.load_log (
    id          SERIAL       PRIMARY KEY,
    source      VARCHAR(100) NOT NULL,
    file_path   TEXT,
    row_count   INT,
    loaded_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    status      VARCHAR(20)  NOT NULL DEFAULT 'success'   -- success | error
);
