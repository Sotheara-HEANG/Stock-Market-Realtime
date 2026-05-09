# econ-governance-pipeline-v2

End-to-end Data Engineering & ML pipeline for economic and governance indicators.
Covers real-time ingestion, PySpark ETL, a three-layer Medallion data warehouse,
MLflow-tracked forecasting, and a FastAPI + Streamlit application layer.

---

## Data Sources

| Source | Indicators | Coverage |
|--------|-----------|----------|
| World Bank WGI | 6 governance scores (corruption, effectiveness, stability, rule of law…) | 1996–2023 |
| IMF World Economic Outlook 2024 | GDP, inflation, unemployment, government debt | 1980–2029 |
| UNDP Human Development Index 2023-24 | HDI, life expectancy, schooling, GNI per capita | 2022 |
| Polity5 | Democracy / autocracy regime scores (−10 to +10) | 1800–2018 |
| V-Dem Core Indices | Electoral, liberal, participatory democracy (0–1) | 1789–2023 |

Optional live source: World Bank WDI REST API (`--api` flag).

---

## Architecture

```
[APIs / CSV Sources]
        │
        ▼
  [Kafka Topics]          ← 01_ingestion/
        │
   ┌────┴────┐
   │         │
   ▼         ▼
PySpark   Pentaho         ← 02_etl/
Streaming  Batch
   │         │
   └────┬────┘
        ▼
┌─────────────────────┐
│   Data Warehouse    │   ← 03_warehouse/
│  Bronze             │   raw staging
│    → Silver         │   cleaned & normalised
│      → Gold         │   star schema (dim/fact)
└─────────────────────┘
        │
        ▼
  [ML Training]           ← 04_ml/
  [MLflow Tracking]
        │
        ▼
[FastAPI + Streamlit]     ← 05_app/
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Ingestion | REST APIs, Kafka Producers (kafka-python) |
| Streaming ETL | PySpark Structured Streaming, Apache Kafka |
| Batch ETL | PySpark Batch, Pentaho Data Integration |
| Data Warehouse | PostgreSQL — Bronze / Silver / Gold (Medallion) |
| ML & Tracking | scikit-learn, statsmodels, MLflow |
| Application | FastAPI, Streamlit, Plotly |
| Containerisation | Docker, Docker Compose |

---

## Project Structure

```
econ-governance-pipeline-v2/
│
├── main.py                        # Legacy batch entry point (Extract → Predict)
├── run_pipeline.sh                # Cron wrapper
├── requirements.txt
├── .env.example
├── docker-compose.yml             # Full stack: Kafka, Postgres, MLflow, API, Dashboard
├── Dockerfile
│
├── 01_ingestion/                  # Step 1 — Data Ingestion
│   ├── api_clients/
│   │   └── source_client.py       # Thin wrappers around ETL extractors
│   ├── kafka_producers/
│   │   └── producer.py            # Publish all sources to Kafka topics
│   └── config/
│       └── sources.yaml           # Kafka config + source schedules
│
├── 02_etl/                        # Step 2 — ETL Pipeline
│   ├── streaming/
│   │   ├── pyspark_streaming/
│   │   │   └── spark_consumer.py  # PySpark Structured Streaming: Kafka → Bronze
│   │   └── kafka/
│   │       └── topics.yaml        # Topic definitions (7 topics)
│   └── batch/
│       ├── pyspark_batch/
│       │   └── batch_job.py       # Batch: Bronze → Silver → Gold
│       └── pentaho/
│           └── jobs/              # Pentaho Kettle .ktr / .kjb jobs
│
├── 03_warehouse/                  # Step 3 — Data Warehouse (Medallion)
│   ├── bronze/
│   │   └── schema.sql             # Raw staging: bronze.raw_indicators
│   ├── silver/
│   │   └── schema.sql             # Cleaned: silver.countries, silver.indicators
│   └── gold/
│       ├── schema.sql             # gold schema
│       ├── dim_country.sql        # Dimension: Country (SCD Type 2)
│       ├── dim_indicator.sql      # Dimension: Indicator (26 pre-populated)
│       ├── dim_time.sql           # Dimension: Time 1960–2030
│       ├── fact_indicators.sql    # Fact: Indicator values
│       └── fact_predictions.sql   # Fact: Model forecasts
│
├── 04_ml/                         # Step 4 — ML Training & Tracking
│   ├── training/
│   │   ├── train.py               # MLflow-tracked training (linear + Holt)
│   │   └── evaluate.py            # MAE, RMSE, MAPE, R² metrics
│   ├── models/                    # Saved prediction CSVs (gitignored)
│   └── mlflow/
│       └── mlflow_config.yaml     # Experiment name, tracking URI, model params
│
├── 05_app/                        # Step 5 — Application
│   ├── api/
│   │   └── app.py                 # FastAPI backend (7 endpoints)
│   └── dashboard/
│       └── dashboard.py           # Streamlit dashboard (5 sections)
│
├── etl/                           # Core ETL modules (shared)
│   ├── extract.py                 # 5 source extractors + World Bank API
│   ├── transform.py               # pandas → PySpark, normalise, pivot
│   ├── enrich.py                  # YoY GDP growth, governance composite, regional avgs
│   ├── load.py                    # Write to PostgreSQL (+ Oracle dual-write)
│   └── predict.py                 # Linear trend + Holt smoothing forecasts
│
├── sql/
│   └── schema.sql                 # Core tables: countries, indicators, predictions
│
├── tests/                         # pytest unit tests
└── Dataset/                       # Source CSV / Excel files
```

---

## Data Warehouse — Medallion Layers

| Layer | Schema | Description |
|---|---|---|
| **Bronze** | `bronze.*` | Raw records from Kafka, no transformation, append-only |
| **Silver** | `silver.*` | Cleaned, normalised, deduplicated, one row per (country, indicator, year) |
| **Gold** | `gold.*` | Star schema — `fact_indicators`, `fact_predictions`, `dim_country`, `dim_indicator`, `dim_time` |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Java 11+ (required by PySpark)
- Docker & Docker Compose

### 1. Clone & install

```bash
git clone https://github.com/KongSattha55/econ-governance-pipeline-v2.git
cd econ-governance-pipeline-v2

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set DB_PASSWORD and optionally ORA_* for Oracle dual-write
```

### 3. Start infrastructure

```bash
# Start Kafka + Zookeeper + PostgreSQL + MLflow
docker compose up -d
```

### 4. Run ingestion (publish to Kafka)

```bash
# Publish all sources
python 01_ingestion/kafka_producers/producer.py

# Publish a single source
python 01_ingestion/kafka_producers/producer.py --source wgi
```

### 5. Run streaming ETL (Kafka → Bronze)

```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,org.postgresql:postgresql:42.6.0 \
  02_etl/streaming/pyspark_streaming/spark_consumer.py
```

### 6. Run batch ETL (Bronze → Silver → Gold)

```bash
spark-submit \
  --packages org.postgresql:postgresql:42.6.0 \
  02_etl/batch/pyspark_batch/batch_job.py
```

### 7. Train models with MLflow tracking

```bash
python 04_ml/training/train.py
# View runs: open http://localhost:5001
```

### 8. Launch the application

```bash
# FastAPI backend  →  http://localhost:8000/docs
uvicorn 05_app.api.app:app --reload

# Streamlit dashboard  →  http://localhost:8501
streamlit run 05_app/dashboard/dashboard.py
```

Or run everything via Docker:

```bash
docker compose --profile app up
```

---

## Streamlit Dashboard Sections

| Section | Description |
|---|---|
| Overview Map | Choropleth of latest indicator values by country |
| Country Deep-Dive | Time-series for any country + indicator |
| Forecasts | Historical data + 3-year model predictions with 95% CI |
| Compare Countries | Overlay multiple countries on one chart |
| Top N Ranking | Bar chart of top / bottom N countries for any indicator + year |

---

## FastAPI Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | DB connectivity check |
| GET | `/countries` | List all countries |
| GET | `/indicators` | List all indicator names |
| GET | `/data/{iso}/{indicator}` | Time-series for one country + indicator |
| GET | `/predictions/{iso}/{indicator}` | Forecasts for one country + indicator |
| GET | `/compare?countries=USA,CHN&indicator=...` | Multi-country comparison |
| GET | `/top?indicator=...&year=2022&n=10` | Top N countries for an indicator |

Interactive docs: `http://localhost:8000/docs`

---

## ML Models

Both models are fitted per `(country, indicator)` series (min 10 observations).
Forecasts cover 3 years beyond the last available data point.

| Model | Description |
|---|---|
| `linear_trend` | OLS trend extrapolation with 95% prediction intervals |
| `holt_smoothing` | Holt double exponential smoothing with damped trend |

MLflow experiment: `econ-governance-forecasting` — tracks parameters, MAE/RMSE per indicator, and prediction artifacts.

---

## Tests

```bash
pytest tests/ -v
```

Covers: all 5 extractors (schema + non-empty output), linear trend and Holt smoothing (output shape + valid confidence intervals).

---

## Example Queries

```sql
-- Top 10 countries by governance composite (latest year)
SELECT c.name, i.year, ROUND(i.value::numeric, 3) AS governance
FROM   silver.indicators i
JOIN   silver.countries  c ON c.id = i.country_id
WHERE  i.indicator = 'governance_composite'
  AND  i.year = (SELECT MAX(year) FROM silver.indicators WHERE indicator = 'governance_composite')
ORDER  BY i.value DESC
LIMIT  10;

-- Gold layer: GDP growth forecast for 2025 (Holt smoothing)
SELECT dc.country_name, fp.predicted_value, fp.confidence_low, fp.confidence_high
FROM   gold.fact_predictions fp
JOIN   gold.dim_country      dc ON dc.country_key   = fp.country_key
JOIN   gold.dim_indicator    di ON di.indicator_key = fp.indicator_key
JOIN   gold.dim_time         dt ON dt.time_key      = fp.time_key
WHERE  di.indicator_name = 'gdp_growth_pct'
  AND  fp.model_name     = 'holt_smoothing'
  AND  dt.year           = 2025
ORDER  BY fp.predicted_value DESC;
```

---

## Pipeline Status

- [x] Step 1 — Data Ingestion (Kafka producers + API clients)
- [x] Step 2 — ETL Pipeline (PySpark Streaming + Batch, Pentaho)
- [x] Step 3 — Data Warehouse (Bronze / Silver / Gold)
- [x] Step 4 — ML Training & Visualization (MLflow)
- [x] Step 5 — Application (FastAPI + Streamlit)
