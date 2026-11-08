#!/usr/bin/env bash
# Compute server (B), daily after the China close: fetch + score China stock news
# and China macro (akshare; mainland sources), then upload them to the production
# server's write-only inbox under china/. A applies the upload at the start of its
# next China daily run. Status goes to status/cn_news.txt for the dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
A="${PROD_SYNC:-fspsync@43.166.232.106}"
LOG="$HOME/cn_news.log"
STATUS_DIR="$HOME/.fsp_status"; mkdir -p "$STATUS_DIR/status"
STARTED="$(date -u +%FT%TZ)"
DATA=markets/china/data

report() {  # report STATE
  local last; last="$( (grep -E 'news progress|CN news|CN macro' "$LOG" 2>/dev/null || true) | tail -1 | cut -c1-200)"
  {
    echo "state=$1"
    echo "started=$STARTED"
    echo "updated=$(date -u +%FT%TZ)"
    echo "last=$last"
    echo "mem=$(free -m | awk '/Mem:/{print $3"/"$2" MB"}')"
  } > "$STATUS_DIR/status/cn_news.txt"
  rsync -a "$STATUS_DIR/status/cn_news.txt" "$A:status/" 2>/dev/null || true
}

: > "$LOG"
( while sleep 300; do report running; done ) & REPORTER=$!
trap 'kill $REPORTER 2>/dev/null; report "${FINAL:-failed}"' EXIT
report running

echo "=== $(date -u +%FT%TZ) china news ===" | tee -a "$LOG"
.venv/bin/python scripts/refresh_cn_news.py 2>&1 | tee -a "$LOG"
echo "=== $(date -u +%FT%TZ) china macro ===" | tee -a "$LOG"
# Macro failure must not block the news upload.
.venv/bin/python scripts/refresh_cn_macro.py 2>&1 | tee -a "$LOG" || echo "china macro failed" | tee -a "$LOG"

echo "=== $(date -u +%FT%TZ) upload to $A inbox ===" | tee -a "$LOG"
rsync -a --delete "$DATA/news/" "$A:china/news/"
[ -d "$DATA/macro" ] && rsync -a "$DATA/macro/" "$A:china/macro/"
[ -f "$DATA/index_compositions/names.parquet" ] && rsync -a "$DATA/index_compositions/names.parquet" "$A:china/names.parquet"
# READY is written last: A only applies an upload whose stamp it has not seen.
date -u +%FT%TZ > /tmp/fsp_cn_READY
rsync -a /tmp/fsp_cn_READY "$A:china/READY"
FINAL=done
echo "=== $(date -u +%FT%TZ) done ===" | tee -a "$LOG"
