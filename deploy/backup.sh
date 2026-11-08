#!/usr/bin/env bash
# Nightly backup of US + China data (parquet, caches, models). Never includes .env.
# China raw OHLCV is skipped (re-downloadable). Local: ~/backups/fsp-data-YYYY-MM-DD.tar.zst, 7 kept.
# Off-site: uploaded to COS when ~/.cos.conf exists (coscli), 7 days kept there.
set -euo pipefail
APP="$HOME/app"; DEST="$HOME/backups"; KEEP=7
mkdir -p "$DEST"
f="$DEST/fsp-data-$(date -u +%F).tar.zst"
parts=(markets/us/data); [ -d "$APP/markets/china/data" ] && parts+=(markets/china/data)
tar -C "$APP" --exclude='*.env' --exclude='.env*' --exclude='markets/china/data/raw' -cf - "${parts[@]}" \n  | zstd -q -T2 -10 -o "$f" -f
ls -1t "$DEST"/fsp-data-*.tar.zst | tail -n +$((KEEP + 1)) | xargs -r rm -f
# Pre-China archives used the old name; drop them once a week of new ones exists.
[ "$(ls -1 "$DEST"/fsp-data-*.tar.zst | wc -l)" -ge "$KEEP" ] && rm -f "$DEST"/us-data-*.tar.zst
echo "local backup: $f ($(du -h "$f" | cut -f1))"

if [ -f "$HOME/.cos.conf" ] && command -v coscli >/dev/null; then
  bucket="$(cat "$HOME/.cos-bucket")"   # e.g. cos://fsp-backup-1250000000
  coscli cp "$f" "$bucket/backups/$(basename "$f")"
  cutoff="$(date -u -d "-$KEEP days" +%F)"
  coscli ls "$bucket/backups/" | grep -oE '(us|fsp)-data-[0-9-]+\.tar\.zst' | sort -u | while read -r name; do
    d="${name#*-data-}"; d="${d%.tar.zst}"
    [[ "$d" < "$cutoff" ]] && coscli rm -f "$bucket/backups/$name"
  done
  echo "uploaded to $bucket"
else
  echo "COS not configured (~/.cos.conf missing): local backup only"
fi
