-- Bronze layer: raw ingested data exactly as received from RapidAPI.
-- One row per (company, indicator) per pipeline run.
-- Never modified after insert — append-only audit trail.

CREATE SCHEMA IF NOT EXISTS bronze;

DROP TABLE IF EXISTS bronze.raw_finance CASCADE;

CREATE TABLE bronze.raw_finance (
    id           SERIAL       PRIMARY KEY,
    company_name VARCHAR(200) NOT NULL,
    country_code VARCHAR(10),
    indicator    VARCHAR(100) NOT NULL,
    year         INT          NOT NULL,
    value        FLOAT,
    source       VARCHAR(100),
    ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_bronze_company   ON bronze.raw_finance (company_name);
CREATE INDEX idx_bronze_indicator ON bronze.raw_finance (indicator);
CREATE INDEX idx_bronze_ingested  ON bronze.raw_finance (ingested_at);
