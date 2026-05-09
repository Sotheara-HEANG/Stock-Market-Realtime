-- Fact Table: Predictions
-- Model forecasts per (country, indicator, future year).

CREATE TABLE IF NOT EXISTS gold.fact_predictions (
    id              BIGSERIAL    PRIMARY KEY,
    country_key     INT          NOT NULL REFERENCES gold.dim_country(country_key),
    indicator_key   INT          NOT NULL REFERENCES gold.dim_indicator(indicator_key),
    time_key        SMALLINT     NOT NULL REFERENCES gold.dim_time(time_key),
    model_name      VARCHAR(100) NOT NULL,
    predicted_value NUMERIC(18, 4),
    confidence_low  NUMERIC(18, 4),
    confidence_high NUMERIC(18, 4),
    -- MLflow tracking
    mlflow_run_id   VARCHAR(50),
    run_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (country_key, indicator_key, time_key, model_name)
);

CREATE INDEX IF NOT EXISTS idx_pred_country   ON gold.fact_predictions(country_key);
CREATE INDEX IF NOT EXISTS idx_pred_indicator ON gold.fact_predictions(indicator_key);
CREATE INDEX IF NOT EXISTS idx_pred_model     ON gold.fact_predictions(model_name);
