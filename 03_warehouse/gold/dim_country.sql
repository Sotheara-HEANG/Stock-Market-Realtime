-- Dimension: Country
-- One row per country. Sourced from silver.countries.

CREATE TABLE IF NOT EXISTS gold.dim_country (
    country_key     SERIAL       PRIMARY KEY,
    iso_code        CHAR(3)      NOT NULL UNIQUE,
    country_name    VARCHAR(150) NOT NULL,
    region          VARCHAR(100),
    income_group    VARCHAR(50),
    -- Continent derived from region for coarser grouping
    continent       VARCHAR(50),
    effective_from  DATE         NOT NULL DEFAULT CURRENT_DATE,
    effective_to    DATE,
    is_current      BOOLEAN      NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_dim_country_iso ON gold.dim_country(iso_code);
