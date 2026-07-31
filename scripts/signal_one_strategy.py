"""Compute live signals for exactly ONE strategy, write to
data/cache/signals_partial/{name}.json, then exit. Run as a fresh subprocess
per strategy (via scripts/run_signals_isolated.py) so memory is fully
released back to the OS between strategies — isolates whether a leak
accumulates across a long-running loop vs. is confined to one strategy.

Usage: python scripts/signal_one_strategy.py <strategy_name>
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from dashboard.config import FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
from scripts.build_features import build_live_features
from src.backtesting.strategy_runner import _is_stateless, _select_cols
from src.features.duckdb_client import load_training_data
from src.strategies.registry import load_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_OUT_DIR = Path("data/cache/signals_partial")
_LIVE_CACHE = Path("data/cache/_tmp_live_features.parquet")
_TRAIN_CACHE = Path("data/cache/_tmp_training_data.parquet")


def _load_live_df() -> pl.DataFrame:
    if _LIVE_CACHE.exists():
        return pl.read_parquet(_LIVE_CACHE)
    return build_live_features()


def _load_train_df() -> pl.DataFrame:
    if _TRAIN_CACHE.exists():
        return pl.read_parquet(_TRAIN_CACHE)
    return load_training_data(PARQUET_DIR)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/signal_one_strategy.py <strategy_name>", file=sys.stderr)
        sys.exit(1)
    name = sys.argv[1]
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== %s: loading data ===", name)
    strategy = load_strategy(name)

    live_df = _load_live_df()
    has_ticker = "ticker" in live_df.columns
    tickers = sorted(live_df["ticker"].unique().to_list()) if has_ticker else ["__all__"]

    rows: list[dict] = []
    try:
        if not _is_stateless(strategy):
            df = _load_train_df()
            train_pd = _select_cols(df, strategy, OHLCV_COLS, FEATURE_COLS).to_pandas()
            if not getattr(strategy, "handles_nan", False):
                train_pd = train_pd.dropna()
            if len(train_pd) < 100:
                logger.warning("%s: too few rows after dropna, skipping", name)
                _write(name, rows)
                return
            logger.info("%s: fitting on %d rows", name, len(train_pd))
            strategy.fit(train_pd)
            del df, train_pd

        for ticker in tickers:
            t_pl = live_df.filter(pl.col("ticker") == ticker) if has_ticker else live_df
            t_pd = _select_cols(t_pl, strategy, OHLCV_COLS, FEATURE_COLS).to_pandas()
            if len(t_pd) == 0 or "time" not in t_pd.columns:
                continue
            try:
                result = strategy.predict(t_pd)
            except Exception as exc:
                logger.debug("%s %s predict failed: %s", name, ticker, exc)
                continue

            pos = len(t_pd) - 1
            if pos >= len(result.signal):
                continue

            sig = str(result.signal.iloc[pos])
            conf = float(result.confidence.iloc[pos])
            close = float(t_pd["close"].iloc[pos]) if "close" in t_pd.columns else 0.0
            ticker_date = t_pd["time"].iloc[pos]
            rows.append({
                "ticker": ticker if ticker != "__all__" else "ALL",
                "date": str(ticker_date),
                "signal": sig,
                "confidence": conf,
                "entry_price": close,
                "position_size": conf,
                "strategy": name,
            })
        logger.info("%s: done — %d signal rows", name, len(rows))
    except Exception as exc:
        logger.error("%s FAILED: %s", name, exc)

    _write(name, rows)


def _write(name: str, rows: list[dict]) -> None:
    out_path = _OUT_DIR / f"{name}.json"
    out_path.write_text(json.dumps({
        "strategy": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": rows,
    }, default=str))
    logger.info("%s: wrote %s", name, out_path)


if __name__ == "__main__":
    main()
