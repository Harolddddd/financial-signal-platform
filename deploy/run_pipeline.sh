#!/usr/bin/env bash
# Server pipeline runner (Linux). Usage: deploy/run_pipeline.sh daily|weekly
# Daily: data + news + macro + features refresh, incremental train, live signals only.
# Weekly: full retrain, then backtests + signals. Same cadence as on the dev PC.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY="$PWD/.venv/bin/python"

step() { echo "=== $(date -u +%FT%TZ) $* ==="; "$PY" "$@"; }

case "${1:-}" in
  daily)
    step scripts/refresh_data.py --market us
    step scripts/incremental_train.py --market us
    step scripts/precompute_dashboard.py --market us --signals-only
    ;;
  weekly)
    step scripts/train_models.py --market us
    step scripts/precompute_dashboard.py --market us
    ;;
  *) echo "usage: $0 daily|weekly" >&2; exit 2 ;;
esac
echo "=== $(date -u +%FT%TZ) $1 pipeline done ==="
