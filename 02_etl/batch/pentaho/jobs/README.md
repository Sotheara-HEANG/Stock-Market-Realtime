# Pentaho Data Integration Jobs

This folder contains Pentaho Kettle `.ktr` (transformations) and `.kjb` (jobs)
for the batch ETL pipeline as an alternative to the PySpark batch job.

## Jobs

| File | Type | Description |
|------|------|-------------|
| `01_extract_sources.kjb` | Job | Orchestrates all extract transformations |
| `02_load_bronze.ktr` | Transformation | Loads raw CSVs into `bronze.raw_indicators` |
| `03_silver_transform.ktr` | Transformation | Cleans and normalises data into `silver.indicators` |
| `04_gold_aggregate.ktr` | Transformation | Builds star-schema tables in `gold.*` |
| `05_full_pipeline.kjb` | Job | Runs all steps end-to-end with error handling |

## Prerequisites

- Pentaho Data Integration (PDI) 9.x / Kettle
- PostgreSQL JDBC driver in `<PDI_HOME>/lib/`
- `.env` variables set (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)

## Running

```bash
# Run the full pipeline job
<PDI_HOME>/kitchen.sh -file=02_etl/batch/pentaho/jobs/05_full_pipeline.kjb \
    -param:DB_HOST=localhost \
    -param:DB_PORT=5432 \
    -param:DB_NAME=econ_pipeline \
    -param:DB_USER=kongsattha \
    -param:DB_PASSWORD=secret \
    -logfile=logs/pentaho_pipeline.log \
    -level=Basic
```

## Notes

- Jobs use the **PostgreSQL** connection defined in the shared DB connection pool.
- All transformations write with `INSERT OR REPLACE` semantics (truncate + insert).
- Error rows are written to `logs/pentaho_errors_<date>.csv` for review.
