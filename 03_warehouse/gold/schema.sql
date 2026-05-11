-- Gold layer: dimensional model ready for analytics and dashboards.
--
-- dim_asset      — one row per unique company/asset
-- fact_prices    — all price metrics per asset per run, FK → dim_asset
-- fact_predictions — ML forecasts per asset per indicator, FK → dim_asset

CREATE SCHEMA IF NOT EXISTS gold;

-- ── Dimension: Asset ────────────────────────────────────────────────────────
DROP TABLE IF EXISTS gold.fact_predictions CASCADE;
DROP TABLE IF EXISTS gold.fact_prices      CASCADE;
DROP TABLE IF EXISTS gold.dim_asset        CASCADE;

CREATE TABLE gold.dim_asset (
    asset_id     SERIAL       PRIMARY KEY,
    company_name VARCHAR(200) NOT NULL UNIQUE,
    sector       VARCHAR(100),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ── Fact: Prices ────────────────────────────────────────────────────────────
CREATE TABLE gold.fact_prices (
    id                    SERIAL  PRIMARY KEY,
    asset_id              INT     NOT NULL REFERENCES gold.dim_asset (asset_id),
    year                  INT     NOT NULL,
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
    loaded_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (asset_id, year)
);

CREATE INDEX idx_gold_prices_asset ON gold.fact_prices (asset_id);
CREATE INDEX idx_gold_prices_year  ON gold.fact_prices (year);

-- ── Fact: Predictions ───────────────────────────────────────────────────────
CREATE TABLE gold.fact_predictions (
    id              SERIAL       PRIMARY KEY,
    asset_id        INT          NOT NULL REFERENCES gold.dim_asset (asset_id),
    indicator       VARCHAR(100) NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    predicted_year  SMALLINT     NOT NULL,
    predicted_value NUMERIC(18,4),
    confidence_low  NUMERIC(18,4),
    confidence_high NUMERIC(18,4),
    run_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (asset_id, indicator, model_name, predicted_year)
);

CREATE INDEX idx_gold_pred_asset ON gold.fact_predictions (asset_id);
CREATE INDEX idx_gold_pred_model ON gold.fact_predictions (model_name);
