-- Silver layer: cleaned, wide-format Finnhub stock price data.
-- One row per ticker symbol per model period.

CREATE SCHEMA IF NOT EXISTS silver;

DROP TABLE IF EXISTS silver.commodity_prices CASCADE;

CREATE TABLE silver.commodity_prices (
    id                  SERIAL       PRIMARY KEY,
    symbol              VARCHAR(40)  NOT NULL,
    commodity_name      VARCHAR(160),
    commodity_category  VARCHAR(80),
    year                INT          NOT NULL,
    open_price          FLOAT,
    high_price          FLOAT,
    low_price           FLOAT,
    close_price         FLOAT,
    latest_price        FLOAT,
    price_change        FLOAT,
    price_change_pct    FLOAT,
    price_trend         VARCHAR(20),
    intraday_range      FLOAT,
    intraday_range_pct  FLOAT,
    volatility_level    VARCHAR(20),
    category_avg_close  FLOAT,
    category_count      INT,
    loaded_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_silver_commodity_symbol ON silver.commodity_prices (symbol);
CREATE INDEX idx_silver_commodity_year   ON silver.commodity_prices (year);
CREATE INDEX idx_silver_commodity_cat    ON silver.commodity_prices (commodity_category);
