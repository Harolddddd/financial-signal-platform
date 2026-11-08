#!/usr/bin/env bash
# Nightly backup of US data (parquet, caches, models). Never includes .env.
# Local: ~/backups/us-data-YYYY-MM-DD.tar.zst, 7 kept.
# Off-site: uploaded to COS when ~/.cos.conf exists (coscli), 7 days kept there.
set -euo pipefail
APP="$HOME/app"; DEST="$HOME/backups"; KEEP=7
mkdir -p "$DEST"
f="$DEST/us-data-$(date -u +%F).tar.zst"
tar -C "$APP" --exclude='*.env' --exclude='.env*' -cf - markets/us/data | zstd -q -T2 -10 -o "$f" -f
ls -1t "$DEST"/us-data-*.tar.zst | tail -n +$((KEEP + 1)) | xargs -r rm -f
echo "local backup: $f ($(du -h "$f" | cut -f1))"

if [ -f "$HOME/.cos.conf" ] && command -v coscli >/dev/null; then
  bucket="$(cat "$HOME/.cos-bucket")"   # e.g. cos://fsp-backup-1250000000
  coscli cp "$f" "$bucket/backups/$(basename "$f")"
  cutoff="$(date -u -d "-$KEEP days" +%F)"
  coscli ls "$bucket/backups/" | grep -oE 'us-data-[0-9-]+\.tar\.zst' | sort -u | while read -r name; do
    d="${name#us-data-}"; d="${d%.tar.zst}"
    [[ "$d" < "$cutoff" ]] && coscli rm -f "$bucket/backups/$name"
  done
  echo "uploaded to $bucket"
else
  echo "COS not configured (~/.cos.conf missing): local backup only"
fi
