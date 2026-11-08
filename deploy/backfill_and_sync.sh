#!/usr/bin/env bash
# Compute server (B): run the FinBERT news backfill, then push the news store
# to the production server (A). A's daily timer is paused during the copy so
# the two never write the news files at the same time.
# Usage: deploy/backfill_and_sync.sh [days]   (default 150)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
A="${PROD_HOST:-ubuntu@43.166.232.106}"
DAYS="${1:-150}"

echo "=== $(date -u +%FT%TZ) backfill ${DAYS} days ==="
.venv/bin/python scripts/backfill_news.py --days "$DAYS"

echo "=== $(date -u +%FT%TZ) sync news to $A ==="
was_active=$(ssh "$A" 'systemctl is-active fsp-daily.timer || true')
ssh "$A" 'sudo systemctl stop fsp-daily.timer; while systemctl is-active -q fsp-daily.service; do sleep 60; done'
restore() { [ "$was_active" = "active" ] && ssh "$A" 'sudo systemctl start fsp-daily.timer' || true; }
trap restore EXIT
rsync -a --delete markets/us/data/news/ "$A:app/markets/us/data/news/"
echo "=== $(date -u +%FT%TZ) done ==="
