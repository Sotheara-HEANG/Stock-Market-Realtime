-- Dimension: Time
-- One row per year. Allows slicing by decade, period, etc.

CREATE TABLE IF NOT EXISTS gold.dim_time (
    time_key    SMALLINT PRIMARY KEY,   -- same as year value
    year        SMALLINT NOT NULL,
    decade      SMALLINT NOT NULL,      -- e.g. 2020 for 2020–2029
    era         VARCHAR(30),            -- Cold War | Post-Cold War | 21st Century
    is_forecast BOOLEAN  NOT NULL DEFAULT FALSE
);

INSERT INTO gold.dim_time (time_key, year, decade, era, is_forecast)
SELECT
    y AS time_key,
    y AS year,
    (y / 10) * 10 AS decade,
    CASE
        WHEN y <= 1991 THEN 'Cold War'
        WHEN y <= 1999 THEN 'Post-Cold War'
        ELSE '21st Century'
    END AS era,
    y > 2024 AS is_forecast
FROM generate_series(1960, 2030) AS y
ON CONFLICT (time_key) DO NOTHING;
