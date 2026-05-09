-- Fact Table: Indicators
-- Central fact table — one row per (country, indicator, year).
-- Foreign keys reference all dimension tables.

CREATE TABLE IF NOT EXISTS gold.fact_indicators (
    id              BIGSERIAL    PRIMARY KEY,
    country_key     INT          NOT NULL REFERENCES gold.dim_country(country_key),
    indicator_key   INT          NOT NULL REFERENCES gold.dim_indicator(indicator_key),
    time_key        SMALLINT     NOT NULL REFERENCES gold.dim_time(time_key),
    value           NUMERIC(18, 4),
    -- Denormalised convenience columns (avoid joins in common queries)
    iso_code        CHAR(3),
    indicator_name  VARCHAR(100),
    source          VARCHAR(100),
    year            SMALLINT,
    loaded_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (country_key, indicator_key, time_key)
);

CREATE INDEX IF NOT EXISTS idx_fact_country   ON gold.fact_indicators(country_key);
CREATE INDEX IF NOT EXISTS idx_fact_indicator ON gold.fact_indicators(indicator_key);
CREATE INDEX IF NOT EXISTS idx_fact_time      ON gold.fact_indicators(time_key);
CREATE INDEX IF NOT EXISTS idx_fact_iso       ON gold.fact_indicators(iso_code);
