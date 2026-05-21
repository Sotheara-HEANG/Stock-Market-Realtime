-- Gold layer: dimensional model for Finnhub stock price analytics.
--
-- dim_commodity          - one row per tracked ticker symbol
-- fact_commodity_prices  - OHLC prices and enrichment metrics per symbol/year
-- fact_predictions       - ML forecasts per ticker/indicator

CREATE SCHEMA IF NOT EXISTS gold;

DROP TABLE IF EXISTS gold.fact_predictions       CASCADE;
DROP TABLE IF EXISTS gold.fact_commodity_prices  CASCADE;
DROP TABLE IF EXISTS gold.dim_commodity          CASCADE;

CREATE TABLE gold.dim_commodity (
    commodity_id       SERIAL       PRIMARY KEY,
    symbol             VARCHAR(40)  NOT NULL UNIQUE,
    commodity_name     VARCHAR(160),
    commodity_category VARCHAR(80),
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE gold.fact_commodity_prices (
    id                  SERIAL PRIMARY KEY,
    commodity_id        INT    NOT NULL REFERENCES gold.dim_commodity (commodity_id),
    year                INT    NOT NULL,
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
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (commodity_id, year)
);

CREATE INDEX idx_gold_commodity_prices_commodity ON gold.fact_commodity_prices (commodity_id);
CREATE INDEX idx_gold_commodity_prices_year      ON gold.fact_commodity_prices (year);

CREATE TABLE gold.fact_predictions (
    id              SERIAL       PRIMARY KEY,
    commodity_id    INT          NOT NULL REFERENCES gold.dim_commodity (commodity_id),
    indicator       VARCHAR(100) NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    predicted_year  SMALLINT     NOT NULL,
    predicted_value NUMERIC(18,4),
    confidence_low  NUMERIC(18,4),
    confidence_high NUMERIC(18,4),
    run_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (commodity_id, indicator, model_name, predicted_year)
);

CREATE INDEX idx_gold_pred_commodity ON gold.fact_predictions (commodity_id);
CREATE INDEX idx_gold_pred_model     ON gold.fact_predictions (model_name);
