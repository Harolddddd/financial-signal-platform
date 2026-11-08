"""Daily paper-trading step (after the signals step): $1000 US account, ¥1000 China account.

Usage: python scripts/paper_trade.py --market us|china
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path

import polars as pl

from config.markets import MARKETS, get_market
from dashboard.data_loader import get_combined_ratings
from src.paper.account import PaperConfig, new_account, run_day

logger = logging.getLogger(__name__)

START_CASH = {"us": 1000.0, "china": 1000.0}
LOT_SIZE = {"us": None, "china": 100}  # US brokers allow fractional shares; A-shares trade in 100s
STATE_FILE = "paper_account.json"


def _bar(raw_dir: Path, ticker: str, day: date) -> dict | None:
    path = raw_dir / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = pl.read_parquet(path, columns=["time", "open", "close"]).filter(pl.col("time").dt.date() == day)
    if d.height == 0:
        return None
    o, c = d["open"][-1], d["close"][-1]
    if o is None or c is None or o != o or c != c or o <= 0:
        return None
    return {"open": float(o), "close": float(c)}


def main(
    market: str,
    raw_dir: Path | None = None,
    cache_dir: Path | None = None,
    benchmark: str | None = None,
    today_override: date | None = None,
) -> dict:
    mcfg = get_market(market)
    raw_dir = raw_dir or mcfg.data_root / "raw" / "ohlcv"
    cache_dir = cache_dir or mcfg.data_root / "cache"
    benchmark = benchmark or mcfg.benchmark_ticker
    path = cache_dir / STATE_FILE
    acct = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
            else new_account(market, mcfg.currency, START_CASH[market]))

    bench = pl.read_parquet(raw_dir / f"{benchmark}.parquet", columns=["time", "close"])
    today = today_override or bench["time"].max().date()

    iso = today.isoformat()
    ratings = [r for r in get_combined_ratings(market)[0] if str(r.get("date", ""))[:10] == iso]
    need = set(acct["positions"]) | {o["ticker"] for o in acct["pending"]} | {r["ticker"] for r in ratings}
    bars = {t: b for t in need if (b := _bar(raw_dir, t, today)) is not None}

    cfg = PaperConfig(cost_bps=mcfg.cost_bps, lot_size=LOT_SIZE[market])
    before = acct.get("last_day")
    acct = run_day(acct, today, bars, ratings, cfg)
    if acct.get("last_day") != before:
        b = bench.filter(pl.col("time").dt.date() == today)["close"]
        acct["equity"][-1]["benchmark"] = float(b[-1]) if len(b) else None

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(acct, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    eq = acct["equity"][-1]["total"] if acct["equity"] else acct["cash"]
    logger.info("paper %s %s: equity %.2f %s, %d positions, %d pending, %d fresh ratings",
                market, iso, eq, acct["currency"], len(acct["positions"]), len(acct["pending"]), len(ratings))
    return acct


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=sorted(MARKETS))
    main(ap.parse_args().market)
