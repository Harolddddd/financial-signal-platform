"""Daily paper-trading step (after the signals step): $2000 US account, ¥10000 China account.

Usage: python scripts/paper_trade.py --market us|china
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.markets import MARKETS, get_market
from dashboard.data_loader import get_combined_ratings
from src.paper.account import PaperConfig, new_account, run_day

logger = logging.getLogger(__name__)

START_CASH = {"us": 2000.0, "china": 10000.0}
LOT_SIZE = {"us": None, "china": 100}  # US brokers allow fractional shares; A-shares trade in 100s
STATE_FILE = "paper_account.json"
_ATR_DAYS = 14
_SELL_MODEL_TRAIN_DAYS = 400


def _bar(raw_dir: Path, ticker: str, day: date) -> dict | None:
    """Today's open/close plus ATR14 from the raw OHLCV file; None without a valid bar."""
    path = raw_dir / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = (pl.read_parquet(path, columns=["time", "open", "high", "low", "close"])
         .filter(pl.col("time").dt.date() <= day).sort("time").tail(_ATR_DAYS + 1))
    if d.height == 0 or d["time"][-1].date() != day:
        return None
    o, c = d["open"][-1], d["close"][-1]
    if o is None or c is None or o != o or c != c or o <= 0:
        return None
    atr = None
    if d.height >= _ATR_DAYS + 1:
        prev = pl.col("close").shift(1)
        tr = d.select(pl.max_horizontal(
            pl.col("high") - pl.col("low"), (pl.col("high") - prev).abs(), (pl.col("low") - prev).abs()
        ).alias("tr"))["tr"]
        v = tr.tail(_ATR_DAYS).mean()
        atr = float(v) if v is not None and v == v and v > 0 else None
    return {"open": float(o), "close": float(c), "atr": atr}


def sell_model_risk(market: str, today: date, tickers: list[str]) -> tuple[set[str], dict]:
    """Train the sell model on the last 400 days and flag today's high-risk tickers."""
    from dashboard.ui_config import FEATURE_COLS
    from scripts.build_features import iter_live_features
    from src.features.duckdb_client import load_training_data
    from src.features.news_macro_features import overlay_cols
    from src.paper.exit_model import ExitModel

    if not tickers:
        return set(), {}
    mcfg = get_market(market)
    start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=_SELL_MODEL_TRAIN_DAYS)
    df = load_training_data(mcfg.data_root / "features", start=start)
    cols = [c for c in FEATURE_COLS + overlay_cols(market) if c in df.columns]
    model = ExitModel(cols).fit(df.select(["time", "ticker", "label", *cols]).to_pandas())
    del df
    live = []
    for ticker, f in iter_live_features(market, tickers=tickers):
        last = f.sort("time").tail(1)
        if last.height and last["time"][0].date() == today:
            live.append(last.select(["ticker", *[c for c in cols if c in last.columns]]))
    rows = pl.concat(live, how="diagonal_relaxed").to_pandas() if live else None
    risky = model.high_risk(rows) if rows is not None else set()
    info = {"trained_on": today.isoformat(), "threshold": model.threshold, "base_rate": model.base_rate,
            "holdout_precision": model.holdout_precision, "checked": len(live), "flagged": sorted(risky)}
    return risky, info


def main(
    market: str,
    raw_dir: Path | None = None,
    cache_dir: Path | None = None,
    benchmark: str | None = None,
    today_override: date | None = None,
    risk_fn=sell_model_risk,
) -> dict:
    from scripts.intraday_guard import account_lock  # the intraday stop check writes the same file

    cache_dir = cache_dir or get_market(market).data_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with account_lock(cache_dir):
        return _run(market, raw_dir, cache_dir, benchmark, today_override, risk_fn)


def _run(market: str, raw_dir: Path | None, cache_dir: Path, benchmark: str | None,
         today_override: date | None, risk_fn) -> dict:
    mcfg = get_market(market)
    raw_dir = raw_dir or mcfg.data_root / "raw" / "ohlcv"
    benchmark = benchmark or mcfg.benchmark_ticker
    path = cache_dir / STATE_FILE
    acct = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
            else new_account(market, mcfg.currency, START_CASH[market]))

    bench = pl.read_parquet(raw_dir / f"{benchmark}.parquet", columns=["time", "close"])
    today = today_override or bench["time"].max().date()
    iso = today.isoformat()
    if acct.get("last_day") and iso <= acct["last_day"]:
        logger.info("paper %s: %s already processed", market, iso)
        return acct

    cfg = PaperConfig(cost_bps=mcfg.cost_bps, lot_size=LOT_SIZE[market])
    ratings = [r for r in get_combined_ratings(market)[0] if str(r.get("date", ""))[:10] == iso]
    candidates = [r["ticker"] for r in sorted(ratings, key=lambda r: -r["overall_rating"])
                  if r["overall_rating"] >= cfg.buy_rating][: 3 * cfg.max_positions]
    need = set(acct["positions"]) | {o["ticker"] for o in acct["pending"]} | {r["ticker"] for r in ratings}
    bars = {t: b for t in need if (b := _bar(raw_dir, t, today)) is not None}

    try:
        risky, info = risk_fn(market, today, sorted(set(acct["positions"]) | set(candidates)))
    except Exception as exc:  # the account still trades on its stop/time rules
        logger.warning("paper %s: sell model unavailable: %s", market, exc)
        risky, info = set(), {"error": str(exc)[:200]}

    acct = run_day(acct, today, bars, ratings, cfg, high_risk=risky)
    acct["sell_model"] = info
    b = bench.filter(pl.col("time").dt.date() == today)["close"]
    acct["equity"][-1]["benchmark"] = float(b[-1]) if len(b) else None

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(acct, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    logger.info("paper %s %s: equity %.2f %s, %d positions, %d pending, %d fresh ratings, %d high-risk",
                market, iso, acct["equity"][-1]["total"], acct["currency"], len(acct["positions"]),
                len(acct["pending"]), len(ratings), len(risky))
    return acct


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=sorted(MARKETS))
    main(ap.parse_args().market)
