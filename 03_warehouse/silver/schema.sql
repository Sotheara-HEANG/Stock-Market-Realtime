-- Silver layer: cleaned, validated, wide-format finance data.
-- One row per company per pipeline run.
-- Derived enrichment columns (intraday_range_pct, price_momentum, sector averages) included.

CREATE SCHEMA IF NOT EXISTS silver;

DROP TABLE IF EXISTS silver.finance_prices CASCADE;

CREATE TABLE silver.finance_prices (
    id                    SERIAL       PRIMARY KEY,
    company_name          VARCHAR(200) NOT NULL,
    sector                VARCHAR(100),
    year                  INT          NOT NULL,
    current_price_usd     FLOAT,
    open_price_usd        FLOAT,
    day_high_usd          FLOAT,
    day_low_usd           FLOAT,
    previous_close_usd    FLOAT,
    price_change_usd      FLOAT,
    price_change_pct      FLOAT,
    trading_volume        FLOAT,
    intraday_range_pct    FLOAT,
    price_momentum        VARCHAR(20),
    sector_avg_price      FLOAT,
    sector_avg_change_pct FLOAT,
    loaded_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_silver_company ON silver.finance_prices (company_name);
CREATE INDEX idx_silver_sector  ON silver.finance_prices (sector);
CREATE INDEX idx_silver_year    ON silver.finance_prices (year);
