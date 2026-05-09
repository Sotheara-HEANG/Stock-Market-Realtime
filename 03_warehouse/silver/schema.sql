-- ============================================================
-- Silver Layer — Cleaned & Validated
-- Normalised country names, deduplicated, typed correctly.
-- One row per (country_code, indicator, year, source).
-- ============================================================

CREATE SCHEMA IF NOT EXISTS silver;

CREATE TABLE IF NOT EXISTS silver.countries (
    id           SERIAL       PRIMARY KEY,
    iso_code     CHAR(3)      NOT NULL UNIQUE,
    name         VARCHAR(150) NOT NULL,
    region       VARCHAR(100),
    income_group VARCHAR(50),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS silver.indicators (
    id           BIGSERIAL    PRIMARY KEY,
    country_id   INT          NOT NULL REFERENCES silver.countries(id) ON DELETE CASCADE,
    indicator    VARCHAR(100) NOT NULL,
    source       VARCHAR(100) NOT NULL,
    year         SMALLINT     NOT NULL,
    value        NUMERIC(18, 4),
    unit         VARCHAR(50),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (country_id, indicator, source, year)
);

-- Derived / computed columns (populated by enrich step)
CREATE TABLE IF NOT EXISTS silver.derived_indicators (
    id                  BIGSERIAL    PRIMARY KEY,
    country_id          INT          NOT NULL REFERENCES silver.countries(id) ON DELETE CASCADE,
    year                SMALLINT     NOT NULL,
    gdp_growth_yoy_calc NUMERIC(10, 4),
    governance_composite NUMERIC(10, 4),
    regional_avg_gdp_growth NUMERIC(10, 4),
    regional_avg_governance NUMERIC(10, 4),
    computed_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (country_id, year)
);

CREATE INDEX IF NOT EXISTS idx_silver_indicators_country   ON silver.indicators(country_id);
CREATE INDEX IF NOT EXISTS idx_silver_indicators_indicator ON silver.indicators(indicator);
CREATE INDEX IF NOT EXISTS idx_silver_indicators_year      ON silver.indicators(year);
CREATE INDEX IF NOT EXISTS idx_silver_derived_country      ON silver.derived_indicators(country_id);
