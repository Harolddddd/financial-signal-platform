#!/usr/bin/env bash
# Server pipeline runner (Linux). Usage: deploy/run_pipeline.sh daily|weekly|signals
# Daily: data + news + macro + features refresh, incremental train, live signals only.
# Weekly: full retrain, then backtests + signals. Same cadence as on the dev PC.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY="$PWD/.venv/bin/python"

step() { echo "=== $(date -u +%FT%TZ) $* ==="; "$PY" "$@"; }

# Apply a news upload from the compute server (B), if there is a new one.
apply_news_inbox() {
  local inbox=/home/fspsync/incoming stamp_file="$PWD/.news_applied"
  [ -f "$inbox/READY" ] && [ -d "$inbox/news" ] || return 0
  local stamp; stamp="$(cat "$inbox/READY")"
  [ "$stamp" = "$(cat "$stamp_file" 2>/dev/null)" ] && return 0
  echo "=== $(date -u +%FT%TZ) applying news upload $stamp ==="
  rsync -a --delete "$inbox/news/" markets/us/data/news/
  echo "$stamp" > "$stamp_file"
}

case "${1:-}" in
  daily)
    apply_news_inbox
    step scripts/refresh_data.py --market us
    step scripts/incremental_train.py --market us
    step scripts/precompute_dashboard.py --market us --signals-only
    ;;
  weekly)
    step scripts/train_models.py --market us
    step scripts/precompute_dashboard.py --market us
    ;;
  signals)
    step scripts/precompute_dashboard.py --market us --signals-only
    ;;
  *) echo "usage: $0 daily|weekly|signals" >&2; exit 2 ;;
esac
echo "=== $(date -u +%FT%TZ) $1 pipeline done ==="
