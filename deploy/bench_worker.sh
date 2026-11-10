#!/usr/bin/env bash
# Compute server (B): pick up a bench request the production server (A) posted,
# run B's share of the methods, and upload the results every quarter.
#
# Downloads use -rt, not -a: copying A's setgid dir bits fails under the unit's RestrictSUIDSGID.
# Two keys, both locked down by rrsync on A's side (fspsync user):
#   ~/.ssh/id_bench_ro -> read-only  /home/fspsync/outgoing  (request, features, code)
#   ~/.ssh/id_ed25519  -> write-only /home/fspsync/incoming  (results go to bench/<market>/<id>/)
# A never logs in to B. Runs every 10 min from fsp-bench-worker.timer; exits at once
# when there is no new request.
set -euo pipefail
A_HOST="${PROD_HOST:-43.166.232.106}"
RO="ssh -i $HOME/.ssh/id_bench_ro -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
export RSYNC_RSH="ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
WORK="$HOME/bench"
PY=/home/fsp/app/.venv/bin/python
mkdir -p "$WORK"
exec 9>"$WORK/.lock"
flock -n 9 || exit 0   # a run is already going

# Phase 1: fetch every new request and its inputs first (the link is slow; A's
# plan counts these downloads before B's first method). Phase 2: run them in
# the order A runs the markets, so B's share of each lands when A expects it.
queue=()
for market in ${BENCH_MARKETS:-us china}; do
  dir="$WORK/$market"
  mkdir -p "$dir"
  rm -f "$dir/request.json"
  rsync -rt -e "$RO" "fspsync@$A_HOST:$market/request.json" "$dir/request.json" 2>/dev/null || continue
  id="$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$dir/request.json")"
  # The id becomes a path on both servers: allow only <market>-<yyyy-mm-dd>-<epoch>.
  [[ "$id" =~ ^${market}-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]+$ ]] || { echo "bad request id: $id" >&2; continue; }
  grep -qxF "$id" "$WORK/done.txt" 2>/dev/null && continue

  echo "=== $(date -u +%FT%TZ) bench request $id ==="
  # Recorded before running: a crash must not restart the job in a loop (A takes it back).
  echo "$id" >> "$WORK/done.txt"
  t0=$(date +%s)
  if rsync -rt --delete -e "$RO" "fspsync@$A_HOST:$market/features/" "$dir/features/" \
     && rsync -rt -e "$RO" "fspsync@$A_HOST:$market/code.tar.gz" "$dir/code.tar.gz"; then
    echo "$market inputs downloaded in $(( $(date +%s) - t0 ))s ($(du -sh "$dir/features" | cut -f1) features)"
    queue+=("$market:$id")
  else
    echo "$market: download failed; A takes this share back" >&2
  fi
done

for item in "${queue[@]}"; do
  market="${item%%:*}"; id="${item#*:}"
  dir="$WORK/$market"
  code="$dir/code"
  rm -rf "$code"; mkdir -p "$code"
  tar xzf "$dir/code.tar.gz" -C "$code"
  out="$dir/out/$id"
  echo "=== $(date -u +%FT%TZ) bench run $id ==="
  (cd "$code" && PYTHONPATH="$code" PYTHONUNBUFFERED=1 "$PY" scripts/bench_worker.py \
      --request "$dir/request.json" --features "$dir/features" --out "$out" \
      --upload-to "fspsync@$A_HOST:bench/$market/$id/") || echo "$market: worker failed; A takes the rest back" >&2
  # Keep the last 3 result sets on B for reference.
  ls -1dt "$dir"/out/*/ 2>/dev/null | tail -n +4 | xargs -r rm -rf
  echo "=== $(date -u +%FT%TZ) bench request $id done ==="
done
