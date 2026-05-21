# Finnhub Stock Streaming Dashboard

This project is a stock market data pipeline and live dashboard.

It collects stock prices from Finnhub, stores the data in PostgreSQL, creates analytics and forecasts, serves the data through FastAPI, and shows a live Streamlit dashboard where stock prices update without refreshing the whole page.

## What This Project Does

1. Gets stock data from Finnhub.
2. Converts the raw data into a clean format.
3. Stores the data in a Bronze, Silver, and Gold warehouse.
4. Builds simple price features like change, trend, range, and volatility.
5. Creates ML forecasts for stock price metrics.
6. Provides an API with FastAPI.
7. Displays a live stock streaming dashboard with Streamlit.

## Main Technologies

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Finnhub API | Stock quote and company data source |
| PySpark | Data transformation |
| PostgreSQL | Data warehouse |
| FastAPI | Backend API |
| Streamlit | Live dashboard UI |
| Plotly | Charts |
| MLflow | ML experiment tracking |
| Docker Compose | Runs all services |

## Data Source

The active data source is Finnhub:

```text
https://finnhub.io/
```

Used endpoints:

```text
GET /quote              live stock price
GET /stock/profile2     company name and sector
GET /stock/candle       historical candle data when the API key allows it
```

Important note: the current Finnhub key can access live quotes. If historical candle access is blocked by the plan, the project uses live quote data plus a projected history fallback so the warehouse and ML pipeline still work.

## Tracked Stocks

Default tickers:

```text
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM
```

You can change them in `.env`:

```bash
FINNHUB_SYMBOLS=AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,JPM
```

## Project Flow

```text
Finnhub API
   |
   v
Extract data
   |
   v
Transform with PySpark
   |
   v
Enrich with price movement, trend, and volatility
   |
   v
Load into PostgreSQL warehouse
   |
   v
Predict future prices and log MLflow runs
   |
   v
FastAPI + Streamlit dashboard
```

## Project Structure

```text
ASSIGMENT2/
|-- main.py                         # Runs the full pipeline
|-- docker-compose.yml              # Defines Postgres, API, dashboard, MLflow
|-- Dockerfile                      # Python app image
|-- requirements.txt                # Python dependencies
|-- .env.example                    # Example environment config
|-- README.md                       # Project guide
|
|-- etl/
|   |-- extract.py                  # Gets data from Finnhub
|   |-- transform.py                # Converts pandas data to Spark and pivots it
|   |-- enrich.py                   # Adds trend, change, range, volatility
|   |-- load.py                     # Writes data to PostgreSQL
|   `-- predict.py                  # Creates forecasts
|
|-- 03_warehouse/
|   |-- bronze/schema.sql           # Raw data table
|   |-- silver/schema.sql           # Clean data table
|   `-- gold/schema.sql             # Analytics and prediction tables
|
|-- 04_ml/
|   |-- mlflow/mlflow_config.yaml   # MLflow settings
|   `-- training/train.py           # MLflow training job
|
|-- 05_app/
|   |-- api/app.py                  # FastAPI backend
|   `-- dashboard/dashboard.py      # Streamlit live dashboard
|
|-- tests/                          # Unit tests
`-- test_realtime.py                # Quick Finnhub test
```

## Warehouse Layers

| Layer | Table | Meaning |
|---|---|---|
| Bronze | `bronze.raw_commodity_prices` | Raw long-format rows from Finnhub |
| Silver | `silver.commodity_prices` | Clean wide stock price rows |
| Gold | `gold.dim_commodity` | Ticker/company/sector dimension |
| Gold | `gold.fact_commodity_prices` | Stock price metrics and enrichments |
| Gold | `gold.fact_predictions` | Forecast results |

The table names still use `commodity_*` because the original project used commodity data. The current data is stock data. In this project:

```text
symbol             = stock ticker
commodity_name     = company name
commodity_category = sector
```

## Dashboard Overview

The Streamlit dashboard shows:

1. Summary cards for average close, average change, rising tickers, falling tickers, and watchlist size.
2. Live market stream table.
3. Up/down status for each stock.
4. Live price, previous close, change, change percent, and small tick movement.
5. Live price chart for all stocks.
6. Stored historical price movement chart.
7. Sector mix chart.

The live stream updates only the live block. It does not refresh the whole page.

## API Overview

FastAPI runs at:

```text
http://localhost:8000
```

Useful endpoints:

| Endpoint | Description |
|---|---|
| `/health` | Check API and database connection |
| `/quote?symbol=AAPL` | Get a live Finnhub quote |
| `/commodities` | List tracked tickers |
| `/prices` | List stock price rows |
| `/prices/AAPL` | Get stored price history for one ticker |
| `/predictions/AAPL` | Get forecast rows for one ticker |
| `/categories` | List sectors |
| `/top?indicator=close_price&n=10` | Rank stocks by a metric |
| `/compare?symbols=AAPL,MSFT` | Compare multiple stocks |

Interactive API docs:

```text
http://localhost:8000/docs
```

## How To Run The Project

### Step 1. Open the project folder

```bash
cd /Users/lychungheang/Documents/ASSIGMENT2
```

### Step 2. Create the `.env` file

If `.env` does not exist:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
DB_HOST=localhost
DB_PORT=55432
DB_NAME=econ_pipeline
DB_USER=Chungheang
DB_PASSWORD=your_database_password

FINNHUB_API_KEY=your_finnhub_api_key
FINNHUB_SYMBOLS=AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,JPM
FINNHUB_HISTORY_YEARS=5
```

Do not commit `.env` because it contains secrets.

### Step 3. Start Docker Desktop

Open Docker Desktop first. Then check Docker:

```bash
docker compose ps
```

If Docker is running, this command should not show a daemon connection error.

### Step 4. Build and start the services

```bash
docker compose --profile app up -d --build postgres api dashboard mlflow
```

This starts:

| Service | URL |
|---|---|
| PostgreSQL | `localhost:55432` |
| FastAPI | `http://localhost:8000` |
| Streamlit dashboard | `http://localhost:8501` |
| MLflow | `http://localhost:5001` |

### Step 5. Run the full pipeline

```bash
docker compose exec api python main.py
```

This runs:

```text
Extract -> Transform -> Enrich -> Load -> Predict -> MLflow Train
```

Use this when you want to rebuild the warehouse and predictions.

### Step 6. Open the dashboard

Open:

```text
http://localhost:8501
```

You should see:

```text
Stock Dashboard -> Live Market Stream
```

The live table updates automatically.

### Step 7. Test the API

Health check:

```bash
curl http://localhost:8000/health
```

Expected result:

```json
{"status":"ok","db":"connected"}
```

Live quote:

```bash
curl "http://localhost:8000/quote?symbol=AAPL"
```

### Step 8. View MLflow

Open:

```text
http://localhost:5001
```

Look for the experiment:

```text
finnhub-stock-forecasting
```

### Step 9. Stop the project

Stop containers but keep database data:

```bash
docker compose --profile app down
```

Stop containers and delete database volumes:

```bash
docker compose --profile app down -v
```

## Useful Commands

Check containers:

```bash
docker compose ps
```

View API logs:

```bash
docker compose logs -f api
```

View dashboard logs:

```bash
docker compose logs -f dashboard
```

Run tests:

```bash
pytest
```

Run Finnhub smoke test:

```bash
python test_realtime.py
```

Run ETL without prediction and MLflow:

```bash
docker compose exec api python main.py --no-predict
```

## Common Problems

### Docker daemon error

If you see:

```text
Cannot connect to the Docker daemon
```

Open Docker Desktop and wait until it is fully running.

### Finnhub candle returns 403

This means the API key does not have access to historical candles. The app still works because it uses live quote projection fallback.

### Dashboard is empty

Run:

```bash
docker compose exec api python main.py
```

Then reload:

```text
http://localhost:8501
```

### API cannot connect to database

Check PostgreSQL:

```bash
docker compose ps
docker compose logs postgres
```

## Short Summary

This project is a complete stock data engineering pipeline:

```text
Finnhub -> PySpark ETL -> PostgreSQL warehouse -> ML predictions -> FastAPI -> live Streamlit dashboard
```
