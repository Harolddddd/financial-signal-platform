#!/usr/bin/env bash
# One-shot status of the server jobs. Usage (from your PC):
#   ssh ubuntu@43.166.232.106 app/deploy/status.sh
echo "== jobs =="
for u in fsp-daily fsp-weekly fsp-backup fsp-dashboard; do
  printf '%-14s %s\n' "$u" "$(systemctl show "$u.service" -p ActiveState -p Result -p ExecMainStartTimestamp --value | paste -sd' ')"
done
echo; echo "== schedule =="
systemctl list-timers 'fsp-*' --no-pager | head -5
echo; echo "== server B backfill =="
cat /home/fspsync/incoming/status/backfill.txt 2>/dev/null || echo "(no status from B yet)"
echo; echo "== resources =="
free -h | sed -n 1,2p; df -h / | tail -1
