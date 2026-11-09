"""Write this server's status snapshot (resources, jobs, timers) as JSON.

Compute server (B) runs it every 5 min via deploy/push_status.sh, which uploads
the file to A's write-only inbox; the dashboard's System Status page reads it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import system_status as ss  # noqa: E402


def main(host: str, out: Path) -> dict:
    services, timers = (ss.SERVICES_B, ss.TIMERS_B) if host == "B" else (ss.SERVICES, ss.TIMERS)
    snap = ss.collect(host, services, [t for t in timers])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap), encoding="utf-8")
    os.replace(tmp, out)
    return snap


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="B", choices=["A", "B"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    main(a.host, a.out)
