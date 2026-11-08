#!/usr/bin/env bash
# Compute server (B): run the FinBERT news backfill, then upload the news store
# to the production server's inbox. B's key on A is restricted to write-only
# rsync into /home/fspsync/incoming; A applies the upload itself at the start
# of its next daily run, so only A ever writes its live news files.
# Usage: deploy/backfill_and_sync.sh [days]   (default 150)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
A="${PROD_SYNC:-fspsync@43.166.232.106}"
DAYS="${1:-150}"

echo "=== $(date -u +%FT%TZ) backfill ${DAYS} days ==="
.venv/bin/python scripts/backfill_news.py --days "$DAYS"

echo "=== $(date -u +%FT%TZ) upload news to $A inbox ==="
rsync -a --delete markets/us/data/news/ "$A:news/"
# READY is written last: A only applies an upload whose stamp it has not seen.
date -u +%FT%TZ > /tmp/fsp_news_READY
rsync -a /tmp/fsp_news_READY "$A:READY"
echo "=== $(date -u +%FT%TZ) done ==="
