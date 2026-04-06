#!/bin/bash
# run_pipeline.sh — monthly wrapper for the economic-governance ETL pipeline
# Scheduled via crontab: 0 1 1 * *  (1st of every month at 01:00 UTC)
#
# To use a specific Python interpreter, set PYTHON_BIN before running:
#   export PYTHON_BIN=/path/to/.venv/bin/python3
# Otherwise the script falls back to "python3" on PATH.

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_BIN:-python3}"
LOG="$PROJECT/logs/pipeline.log"

mkdir -p "$PROJECT/logs"
cd "$PROJECT" || exit 1

echo "======================================" >> "$LOG"
echo "Run started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
echo "======================================" >> "$LOG"

"$PYTHON" main.py >> "$LOG" 2>&1
EXIT_CODE=$?

echo "Exit code: $EXIT_CODE" >> "$LOG"
echo "Run finished: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
echo "" >> "$LOG"

exit $EXIT_CODE
