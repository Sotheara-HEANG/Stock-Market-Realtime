# econ-etl-pipeline

End-to-end ETL + forecasting pipeline for economic and governance indicators.
Pulls data from 5 public datasets, merges them with PySpark, loads into PostgreSQL,
and fits time-series models to forecast key indicators 3 years ahead.

## Data Sources

| Source | Indicators | Coverage |
|--------|-----------|----------|
| World Bank WGI | 6 governance scores (corruption, effectiveness, stability, rule of law…) | 1996–2023 |
| IMF World Economic Outlook 2024 | GDP, inflation, unemployment, government debt | 1980–2029 |
| UNDP Human Development Index 2023-24 | HDI, life expectancy, schooling, GNI per capita | 2022 |
| Polity5 | Democracy / autocracy regime scores (−10 to +10) | 1800–2018 |
| V-Dem Core Indices | Electoral, liberal, participatory democracy (0–1) | 1789–2023 |

Optional live source: World Bank WDI REST API (`--api` flag).

## Pipeline Stages

```
Extract → Transform → Enrich → Load → Predict
```

1. **Extract** (`etl/extract.py`) — reads each source CSV into a tidy long-format pandas DataFrame with a consistent 6-column schema: `country_code, country_name, indicator, year, value, source`
2. **Transform** (`etl/transform.py`) — converts to PySpark, normalizes country names across 70+ naming conflicts, pivots to wide format (one row per country + year), drops rows missing both GDP and HDI
3. **Enrich** (`etl/enrich.py`) — adds year-over-year GDP growth (from GDP levels via window lag), governance composite score (mean of ≥3 WGI indicators), and per-(continent, year) regional averages
4. **Load** (`etl/load.py`) — writes `countries` and `indicators` tables to PostgreSQL; clears and rewrites on every run for idempotency
5. **Predict** (`etl/predict.py`) — fits **linear trend** (OLS with 95% prediction intervals) and **Holt double exponential smoothing** per (country, indicator) series; writes to `predictions` table; requires ≥10 observations per series

## Database Schema

```
countries    — master list     (id, iso_code, name, region)
indicators   — time-series     (country_id → indicator, source, year, value, unit)
predictions  — model output    (country_id → indicator, model_name, predicted_year, value, CI)
```

Full DDL: [`sql/schema.sql`](sql/schema.sql)

## Setup

### Prerequisites

- Python 3.10+
- Java 11+ (required by PySpark)
- PostgreSQL 14+

### Install

```bash
git clone https://github.com/KongSattha55/econ-etl-pipeline.git
cd econ-etl-pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure database

```bash
cp .env.example .env             # fill in your PostgreSQL credentials
psql -d econ_pipeline -f sql/schema.sql
```

### Run

```bash
# Full pipeline (offline sources only)
python main.py

# Include live World Bank API data
python main.py --api

# Skip the prediction step
python main.py --no-predict
```

### Monthly automation (cron)

```bash
chmod +x run_pipeline.sh
# Optionally set your Python path:  export PYTHON_BIN=/path/to/python3
crontab -e
# Add: 0 1 1 * * /path/to/run_pipeline.sh
```

## Project Structure

```
econ-etl-pipeline/
├── main.py                  # pipeline entry point (Extract → … → Predict)
├── run_pipeline.sh          # cron wrapper, logs to logs/pipeline.log
├── requirements.txt
├── .env.example             # copy to .env and fill in DB credentials
│
├── etl/
│   ├── extract.py           # 6 source extractors (WGI, IMF, HDI, Polity5, V-Dem, WB API)
│   ├── transform.py         # pandas → Spark, country name normalization, pivot
│   ├── enrich.py            # YoY GDP growth, governance composite, regional averages
│   ├── load.py              # writes countries + indicators to PostgreSQL
│   ├── predict.py           # linear trend + Holt smoothing forecasts
│   └── raw.py               # saves raw source files as parquet snapshots
│
├── sql/
│   └── schema.sql           # PostgreSQL DDL (countries, indicators, predictions)
│
├── Dataset/                 # source CSV / Excel files (not modified by pipeline)
└── tests/                   # pytest unit tests
```

## Tests

```bash
pytest tests/ -v
```

Tests cover:
- All 5 extractors produce the correct 6-column schema with non-empty output
- Linear trend and Holt smoothing models produce correct output shape and valid confidence intervals

## Example Queries

```sql
-- Top 10 countries by governance composite (latest year available)
SELECT c.name, i.year, ROUND(i.value::numeric, 3) AS governance
FROM   indicators i
JOIN   countries  c ON c.id = i.country_id
WHERE  i.indicator = 'governance_composite'
  AND  i.year = (SELECT MAX(year) FROM indicators WHERE indicator = 'governance_composite')
ORDER  BY i.value DESC
LIMIT  10;

-- GDP growth forecast for 2025 using Holt smoothing
SELECT c.name,
       p.predicted_value,
       p.confidence_low,
       p.confidence_high
FROM   predictions p
JOIN   countries   c ON c.id = p.country_id
WHERE  p.indicator    = 'gdp_growth_pct'
  AND  p.model_name   = 'holt_smoothing'
  AND  p.predicted_year = 2025
ORDER  BY p.predicted_value DESC;
```
