#!/usr/bin/env bash
# Compute server (B), every 5 min: snapshot B's resources + jobs and upload to A's
# write-only inbox (status/server_b.json) for the dashboard's System Status page.
set -euo pipefail
cd "$(dirname "$0")/.."
A="${PROD_SYNC:-fspsync@43.166.232.106}"
OUT="$HOME/.fsp_status/status/server_b.json"
PYTHONPATH="$PWD" .venv/bin/python scripts/server_status.py --host B --out "$OUT"
rsync -rt --timeout=60 -e "ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20" \
  "$OUT" "$A:status/server_b.json"
