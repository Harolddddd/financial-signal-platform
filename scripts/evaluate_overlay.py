"""Compare plain vs news/macro-overlay strategies on a walk-forward backtest (US).

Usage: python scripts/evaluate_overlay.py --since YYYY-MM-DD
Writes markets/us/data/cache/overlay_eval.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.markets import get_market
from dashboard.ui_config import FEATURE_COLS, OHLCV_COLS
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.features.news_macro_features import OVERLAY_COLS
from src.signals.overlay import OverlayConfig, OverlayStrategy
from src.strategies.registry import load_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 400


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("  wrote %s", path)


def evaluate(
    df: pl.DataFrame,
    names: list[str],
    cfg: OverlayConfig,
    **wf_kwargs,
) -> list[dict]:
    """Walk-forward each strategy plain and with the overlay; one row per variant.

    A strategy that raises is logged and skipped (no rows).
    """
    wf_kwargs.setdefault("train_window_days", 400)
    wf_kwargs.setdefault("test_window_days", 21)
    wf_kwargs.setdefault("step_days", 21)
    rows: list[dict] = []
    for name in names:
        logger.info("  strategy: %s", name)
        try:
            variants = []
            for variant, feature_cols in (
                ("plain", FEATURE_COLS),
                ("overlay", FEATURE_COLS + OVERLAY_COLS),
            ):
                strategy = load_strategy(name)
                if variant == "overlay":
                    strategy = OverlayStrategy(strategy, cfg)
                wf = walk_forward_backtest_strategy(
                    df, strategy, OHLCV_COLS, feature_cols, **wf_kwargs,
                )
                variants.append({
                    "strategy": name,
                    "variant": variant,
                    "n_trades": sum(f.n_trades for f in wf.folds),
                    "sharpe": wf.mean_sharpe,
                    "precision_buy": wf.weighted_precision_buy,
                })
        except Exception as exc:
            logger.error("  FAILED %s: %s", name, exc)
            continue
        rows.extend(variants)
    return rows


def _log_table(rows: list[dict]) -> None:
    logger.info("%-28s %-8s %8s %8s %8s", "strategy", "variant", "trades", "sharpe", "prec_buy")
    for r in rows:
        logger.info("%-28s %-8s %8d %8.3f %8.3f",
                    r["strategy"], r["variant"], r["n_trades"], r["sharpe"], r["precision_buy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, help="YYYY-MM-DD start of the news window")
    args = parser.parse_args()
    since = date.fromisoformat(args.since)

    market = get_market("us")
    cfg = OverlayConfig()
    df = load_training_data(market.data_root / "features")
    df = df.filter(pl.col("time") >= since - timedelta(days=LOOKBACK_DAYS))
    rows = evaluate(df, list(market.default_strategies), cfg)
    _log_table(rows)
    _write(market.data_root / "cache" / "overlay_eval.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "since": args.since,
        "config": dataclasses.asdict(cfg),
        "rows": rows,
    })


if __name__ == "__main__":
    main()
