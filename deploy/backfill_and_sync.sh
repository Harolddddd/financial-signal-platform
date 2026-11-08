#!/usr/bin/env bash
# Compute server (B): run the FinBERT news backfill, then upload the news store
# to the production server's inbox. B's key on A is restricted to write-only
# rsync into /home/fspsync/incoming; A applies the upload itself at the start
# of its next daily run, so only A ever writes its live news files.
# Progress is uploaded to the same inbox (status/backfill.txt) every 5 minutes
# for the dashboard's System Status page.
# Usage: deploy/backfill_and_sync.sh [days]   (default 150)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
A="${PROD_SYNC:-fspsync@43.166.232.106}"
DAYS="${1:-150}"
LOG="$HOME/backfill.log"
STATUS_DIR="$HOME/.fsp_status"; mkdir -p "$STATUS_DIR/status"
STARTED="$(date -u +%FT%TZ)"

report() {  # report STATE
  local last; last="$(grep -E 'news progress|News backfill' "$LOG" 2>/dev/null | tail -1 | cut -c1-200)"
  {
    echo "state=$1"
    echo "days=$DAYS"
    echo "started=$STARTED"
    echo "updated=$(date -u +%FT%TZ)"
    echo "last=$last"
    echo "mem=$(free -m | awk '/Mem:/{print $3"/"$2" MB"}')"
  } > "$STATUS_DIR/status/backfill.txt"
  rsync -a "$STATUS_DIR/status/" "$A:status/" 2>/dev/null || true
}

: > "$LOG"
( while sleep 300; do report running; done ) & REPORTER=$!
trap 'kill $REPORTER 2>/dev/null; report "${FINAL:-failed}"' EXIT
report running

echo "=== $(date -u +%FT%TZ) backfill ${DAYS} days ===" | tee -a "$LOG"
.venv/bin/python scripts/backfill_news.py --days "$DAYS" 2>&1 | tee -a "$LOG"

echo "=== $(date -u +%FT%TZ) upload news to $A inbox ===" | tee -a "$LOG"
rsync -a --delete markets/us/data/news/ "$A:news/"
# READY is written last: A only applies an upload whose stamp it has not seen.
date -u +%FT%TZ > /tmp/fsp_news_READY
rsync -a /tmp/fsp_news_READY "$A:READY"
FINAL=done
echo "=== $(date -u +%FT%TZ) done ===" | tee -a "$LOG"
