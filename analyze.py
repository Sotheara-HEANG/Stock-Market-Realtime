"""
analyze.py — Extract from the database via Spark JDBC, then run EDA and aggregations.

This is the inverse of main.py: instead of loading data INTO the DB,
this script pulls the stored data OUT and analyses it with Spark.

Usage:
    python analyze.py                          # PostgreSQL (default)
    python analyze.py --db mysql               # MySQL / MariaDB
    python analyze.py --db sqlserver           # SQL Server
    python analyze.py --save                   # save aggregations to data/aggregations/
    python analyze.py --skip-eda               # skip EDA, run aggregations only
    python analyze.py --indicator gdp_usd_bn   # change the sample indicator shown

Pipeline:
    1. get_spark_jdbc()      — SparkSession with JDBC driver (auto-downloaded)
    2. extract_from_db()     — read countries + indicators via JDBC, join + cache
    3. run_eda()             — null profile, stats, year/country/region coverage
    4. aggregate()           — regional, global trend, top countries, YoY change
"""

from __future__ import annotations

import sys
from pathlib import Path

from etl.spark_db_extract import get_spark_jdbc, extract_from_db
from etl.spark_eda        import run_eda
from etl.spark_aggregate  import aggregate


def _parse_args(argv: list[str]) -> dict:
    args = {
        "db_type":         "postgresql",
        "save":            False,
        "skip_eda":        False,
        "sample_indicator": "gdp_growth_pct",
        "sample_year_from": 2015,
    }
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--db" and i + 1 < len(argv):
            args["db_type"] = argv[i + 1]
            i += 2
        elif token.startswith("--db="):
            args["db_type"] = token.split("=", 1)[1]
            i += 1
        elif token == "--save":
            args["save"] = True
            i += 1
        elif token == "--skip-eda":
            args["skip_eda"] = True
            i += 1
        elif token == "--indicator" and i + 1 < len(argv):
            args["sample_indicator"] = argv[i + 1]
            i += 2
        elif token.startswith("--indicator="):
            args["sample_indicator"] = token.split("=", 1)[1]
            i += 1
        else:
            i += 1
    return args


if __name__ == "__main__":
    cfg = _parse_args(sys.argv[1:])

    out_dir = (
        str(Path(__file__).parent / "data" / "aggregations")
        if cfg["save"] else None
    )

    # -----------------------------------------------------------------------
    # Step 1 — Connect + extract from database
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"Step 1/3 — Extract from {cfg['db_type']}")
    print("=" * 60)
    spark = get_spark_jdbc(cfg["db_type"])
    df = extract_from_db(spark, db_type=cfg["db_type"])

    # -----------------------------------------------------------------------
    # Step 2 — EDA
    # -----------------------------------------------------------------------
    if not cfg["skip_eda"]:
        print("\n" + "=" * 60)
        print("Step 2/3 — EDA")
        print("=" * 60)
        eda_results = run_eda(df)
    else:
        print("\nStep 2/3 — EDA  [skipped via --skip-eda]")

    # -----------------------------------------------------------------------
    # Step 3 — Aggregations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3/3 — Aggregate")
    print("=" * 60)
    agg_results = aggregate(
        df,
        out_dir=out_dir,
        sample_indicator=cfg["sample_indicator"],
        sample_year_from=cfg["sample_year_from"],
    )

    spark.stop()
    print("\n" + "=" * 60)
    print("Done.")
    if out_dir:
        print(f"Aggregations saved to: {out_dir}/")
    print("=" * 60)
