"""Bench re-test: every non-retired method of a market on the last 3 years, then league update.

Runs from the weekly Sun 13:00 ET timer; exits at once unless the last completed
pass is >= bench_min_gap_days old (so it really runs every 2nd week), or --force.

Usage: python scripts/bench_pass.py --market us [--deadline ISO] [--force] [--seed-from-pass]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.markets import MARKETS, get_market
from dashboard.ui_config import FEATURE_COLS, OHLCV_COLS
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.league import league as lg
from src.league.scoring import score_predictions
from src.strategies.registry import list_strategies, load_strategy

warnings.filterwarnings("ignore", message=r"`sklearn\.utils\.parallel\.delayed` should be used")
logger = logging.getLogger(__name__)

_TRAIN_WINDOW_DAYS = 400
_PRED_SCHEMA = {"ticker": pl.Utf8, "time": pl.Date, "buy_score": pl.Float64}


def test_order(names: list[str], state: dict) -> list[str]:
    """Never-tested (or stopped-before) methods first, then the oldest test date."""
    def last(n: str) -> str:
        m = state["methods"].get(n)
        return m["history"][-1]["date"] if m and m["history"] else ""
    return sorted(names, key=last)


def _safe(name: str) -> str:
    return name.replace("/", "_")


def main(
    market: str,
    deadline: datetime | None = None,
    strategies: list[str] | None = None,
    force: bool = False,
    seed_from_pass: bool = False,
    today: date | None = None,
    feature_dir: Path | None = None,
    cache_dir: Path | None = None,
    load=load_training_data,
    loader=load_strategy,
) -> dict:
    cfg = lg.load_config()
    mcfg = get_market(market)
    feature_dir = feature_dir or mcfg.data_root / "features"
    cache_dir = cache_dir or mcfg.data_root / "cache"
    today = today or datetime.now(timezone.utc).date()

    state = lg.load_state(cache_dir, market, seed=list(mcfg.default_strategies or ()))
    last = lg.last_completed_pass(state)
    if not force and last is not None and (today - last).days < cfg["bench_min_gap_days"]:
        logger.info("bench %s: last pass %s, skipping", market, last)
        return {"skipped": "recent"}

    retired = {n for n, m in state["methods"].items() if m["status"] == "retired"}
    names = test_order([n for n in (strategies or list_strategies(market)) if n not in retired], state)

    window = cfg["bench_window_days"]
    t0 = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    test_start = t0 - timedelta(days=window)
    df = load(feature_dir, start=test_start - timedelta(days=_TRAIN_WINDOW_DAYS))
    outcomes = df.select("ticker", pl.col("time").dt.date().alias("time"), "forward_return_5d")
    bench_dir = cache_dir / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    preds: dict[str, pl.DataFrame] = {}
    stopped = False
    for i, name in enumerate(names, start=1):
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            logger.warning("bench %s: deadline reached, %d methods left", market, len(names) - i + 1)
            stopped = True
            break
        logger.info("bench %s %d/%d: %s", market, i, len(names), name)
        try:
            log: list = []
            wf = walk_forward_backtest_strategy(
                df, loader(name), OHLCV_COLS, FEATURE_COLS,
                train_window_days=_TRAIN_WINDOW_DAYS, test_window_days=21, step_days=21,
                start=test_start, pred_log=log,
            )
            p = pl.DataFrame(log, schema=_PRED_SCHEMA, orient="row")
            score = score_predictions(p, outcomes, cost_bps=mcfg.cost_bps)
        except Exception as exc:
            logger.error("bench %s: %s failed: %s", market, name, exc)
            results[name] = {"error": str(exc)[:300]}
            continue
        results[name], preds[name] = score, p
        p.write_parquet(bench_dir / f"preds_{_safe(name)}.parquet")
        (bench_dir / f"bench_{_safe(name)}.json").write_text(json.dumps({
            "strategy_name": name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tested_on": today.isoformat(),
            "window_days": window,
            "score": score,
            "mean_sharpe": wf.mean_sharpe,
            "weighted_precision_buy": wf.weighted_precision_buy,
            "worst_drawdown": wf.worst_drawdown,
            "n_folds": len(wf.folds),
        }, indent=2), encoding="utf-8")
        logger.info("  %s net_sharpe=%.2f trades=%d", name, score["net_sharpe"], score["n_trades"])

    lg.update_league(state, results, preds, today, cfg, completed=not stopped)
    if seed_from_pass:
        lg.seed_live_from_latest(state, cfg, n=cfg["live_max"] - 1)
    retired_dir = cache_dir / "retired"
    for n, m in state["methods"].items():
        src = bench_dir / f"bench_{_safe(n)}.json"
        if m["status"] == "retired" and src.exists():
            retired_dir.mkdir(exist_ok=True)
            shutil.move(str(src), retired_dir / src.name)
    lg.save_state(cache_dir, state)

    live = [n for n, m in state["methods"].items() if m["status"] == "live"]
    logger.info("bench %s done: live=%s", market, live)
    return {
        "tested": [n for n in results if "error" not in results[n]],
        "errors": [n for n in results if "error" in results[n]],
        "stopped": stopped,
        "live": live,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us", choices=sorted(MARKETS))
    ap.add_argument("--deadline", type=datetime.fromisoformat)
    ap.add_argument("--strategies", type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed-from-pass", action="store_true")
    a = ap.parse_args()
    main(a.market, deadline=a.deadline, strategies=a.strategies, force=a.force, seed_from_pass=a.seed_from_pass)
