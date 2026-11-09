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

# Apply a China news + macro upload from B (inbox china/), if there is a new one.
apply_cn_inbox() {
  local inbox=/home/fspsync/incoming/china stamp_file="$PWD/.cn_applied"
  [ -f "$inbox/READY" ] || return 0
  local stamp; stamp="$(cat "$inbox/READY")"
  [ "$stamp" = "$(cat "$stamp_file" 2>/dev/null)" ] && return 0
  echo "=== $(date -u +%FT%TZ) applying china upload $stamp ==="
  mkdir -p markets/china/data/news markets/china/data/macro markets/china/data/index_compositions
  [ -d "$inbox/news" ] && rsync -a --delete "$inbox/news/" markets/china/data/news/
  [ -d "$inbox/macro" ] && rsync -a "$inbox/macro/" markets/china/data/macro/
  [ -f "$inbox/names.parquet" ] && cp "$inbox/names.parquet" markets/china/data/index_compositions/names.parquet
  echo "$stamp" > "$stamp_file"
}

# Bench hard stop: the earlier of next Monday 04:30 New York time and 15 hours
# from now (a Monday catch-up run must not get a deadline a week away).
# Prints epoch seconds.
bench_deadline() {
  local mon cap
  mon="$(TZ=America/New_York date -d "next monday 04:30" +%s)"
  cap="$(date -d "+15 hours" +%s)"
  echo $(( mon < cap ? mon : cap ))
}
iso() { date -u -d "@$1" +%FT%T+00:00; }

# One heavy job at a time on this 8 GB server: a job that starts while another
# runs (e.g. China weekly still going when the bench pass fires) waits here.
exec 9>/tmp/fsp-heavy.lock
flock 9

case "${1:-}" in
  daily)
    apply_news_inbox
    step scripts/refresh_data.py --market us
    step scripts/incremental_train.py --market us
    step scripts/precompute_dashboard.py --market us --signals-only
    step scripts/paper_trade.py --market us
    ;;
  weekly)
    step scripts/train_models.py --market us
    step scripts/precompute_dashboard.py --market us
    ;;
  signals)
    step scripts/precompute_dashboard.py --market us --signals-only
    ;;
  cn-daily)
    apply_cn_inbox
    step scripts/refresh_data.py --market china
    step scripts/incremental_train.py --market china
    step scripts/precompute_dashboard.py --market china --signals-only
    step scripts/paper_trade.py --market china
    ;;
  cn-weekly)
    step scripts/train_models.py --market china
    step scripts/precompute_dashboard.py --market china
    ;;
  bench)
    # US gets the first half of the time, China the rest; a US failure must
    # not skip China, and the job still reports failure at the end.
    end="$(bench_deadline)"; now="$(date +%s)"; mid=$(( now + (end - now) / 2 ))
    rc=0
    # Post China's share for the compute server first, so B works on it while
    # A runs the US (no-op unless remote_bench.china.enabled; failure = A does all).
    step scripts/bench_pass.py --market china --publish-only --head-start-market us --deadline "$(iso "$end")" || echo "bench china: posting B's share failed, A runs it all"
    step scripts/bench_pass.py --market us --deadline "$(iso "$mid")" || { echo "bench us FAILED"; rc=1; }
    step scripts/bench_pass.py --market china --deadline "$(iso "$end")" || { echo "bench china FAILED"; rc=1; }
    [ "$rc" -eq 0 ] || exit "$rc"
    ;;
  *) echo "usage: $0 daily|weekly|signals|cn-daily|cn-weekly|bench" >&2; exit 2 ;;
esac
echo "=== $(date -u +%FT%TZ) $1 pipeline done ==="
