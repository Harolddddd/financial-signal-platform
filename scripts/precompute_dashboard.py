"""
Run locally to pre-compute all dashboard data and write JSON to the given
market's markets/<market>/data/cache/ (default: us).
Commit markets/<market>/data/cache/ and push — Render reads from these files instead of
recomputing on every cold start.

Usage:
    python scripts/precompute_dashboard.py [--market us|china]
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from config.markets import MARKETS, get_market
from dashboard.ui_config import CONFIDENCE_THRESHOLD, FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
from scripts.build_features import iter_live_features
from src.backtesting.grader import grade_model
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.strategy_runner import _is_stateless, _select_cols, walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.strategies.registry import list_strategies, load_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = get_market("us").data_root / "cache"


def _market_paths(market: str) -> tuple[Path, Path]:
    market_cfg = get_market(market)
    return market_cfg.data_root / "features", market_cfg.data_root / "cache"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metrics_dict(m) -> dict:
    return {
        "n_trades": m.n_trades,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "total_return_pct": m.total_return_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "max_drawdown_pct": m.max_drawdown_pct,
        "precision_buy": m.precision_buy,
        "recall_buy": m.recall_buy,
        "f1_buy": m.f1_buy,
        "accuracy": m.accuracy,
    }


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("  wrote %s", path)


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _at(seq, i: int):
    # Positional access that works for plain lists and pandas Series alike
    # (a Series' `[]` is label-based and would misbehave on a non-default index).
    return seq.iloc[i] if hasattr(seq, "iloc") else seq[i]


def _trade_window(signals, dates, pos: int) -> dict:
    """Trade lifecycle for a signal series ending at `pos` (inclusive).

    "open" — currently in a Buy run; open_date is where that run started.
    "closed" — most recent Buy run has ended; close_date is the first
      non-Buy date after it (i.e. when the position would have been exited).
    "none" — no Buy signal anywhere at or before `pos`.
    """
    current = str(_at(signals, pos))
    if current == "Buy":
        start = pos
        while start > 0 and str(_at(signals, start - 1)) == "Buy":
            start -= 1
        return {"status": "open", "open_date": str(_at(dates, start)), "close_date": None}

    end = pos - 1
    while end >= 0 and str(_at(signals, end)) != "Buy":
        end -= 1
    if end < 0:
        return {"status": "none", "open_date": None, "close_date": None}

    start = end
    while start > 0 and str(_at(signals, start - 1)) == "Buy":
        start -= 1
    close_idx = end + 1
    return {"status": "closed", "open_date": str(_at(dates, start)), "close_date": str(_at(dates, close_idx))}


# ---------------------------------------------------------------------------

def step_data_summary(feature_dir: Path = PARQUET_DIR, cache_dir: Path = CACHE_DIR) -> None:
    logger.info("[1/4] data summary")
    # Read directly from parquets — never read from cache here, that defeats the purpose.
    df = load_training_data(feature_dir)
    tickers = sorted(df["ticker"].unique().to_list()) if "ticker" in df.columns else []
    summary = {
        "n_tickers": len(tickers),
        "n_rows": len(df),
        "tickers": tickers,
        "date_range_start": str(df["time"].min()) if "time" in df.columns else "N/A",
        "date_range_end":   str(df["time"].max()) if "time" in df.columns else "N/A",
        "generated_at": _now(),
    }
    _write(cache_dir / "data_summary.json", summary)


def step_leaderboard(cache_dir: Path = CACHE_DIR) -> None:
    logger.info("[3/4] leaderboard — aggregating from per-strategy backtest caches")
    # Build the leaderboard from the individual backtest files written in step_backtests().
    # Never call get_leaderboard() here — it reads the old leaderboard.json and writes it back.
    grades: list[dict] = []
    for name in list_strategies():
        cache_path = cache_dir / f"backtest_{_safe(name)}.json"
        if cache_path.exists():
            d = json.loads(cache_path.read_text())
            grades.append(d["grade"])
        else:
            logger.warning("  no backtest cache for %s — skipping from leaderboard", name)

    grades.sort(key=lambda g: g["composite_score"], reverse=True)
    _write(cache_dir / "leaderboard.json", {
        "generated_at": _now(),
        "grades": grades,
    })


def step_backtests(
    names: list[str] | None = None,
    step_days: int = 21,
    feature_dir: Path = PARQUET_DIR,
    cache_dir: Path = CACHE_DIR,
) -> None:
    logger.info("[2/4] per-strategy backtests — running walk-forward fresh (no cache read)")
    # Load data once; reuse across all strategies.
    df = load_training_data(feature_dir)
    for name in (names if names is not None else list_strategies()):
        logger.info("  strategy: %s", name)
        try:
            strategy = load_strategy(name)
            wf = walk_forward_backtest_strategy(
                df, strategy, OHLCV_COLS, FEATURE_COLS,
                train_window_days=400, test_window_days=21, step_days=step_days,
            )
            # Aggregate across all folds for the grade.
            total_trades = sum(f.n_trades for f in wf.folds)
            avg_metrics = BacktestMetrics(
                n_trades=total_trades,
                win_rate=wf.mean_win_rate,
                profit_factor=0.0,
                total_return_pct=0.0,
                sharpe_ratio=wf.mean_sharpe,
                max_drawdown_pct=wf.worst_drawdown,
                precision_buy=wf.mean_precision_buy,
                recall_buy=0.0,
                f1_buy=0.0,
                accuracy=0.0,
            )
            g = grade_model(name, avg_metrics)
            _write(cache_dir / f"backtest_{_safe(name)}.json", {
                "generated_at": _now(),
                "strategy_name": name,
                "mean_sharpe": wf.mean_sharpe,
                "mean_win_rate": wf.mean_win_rate,
                "mean_precision_buy": wf.mean_precision_buy,
                "worst_drawdown": wf.worst_drawdown,
                "grade": {
                    "model_name": g.model_name,
                    "grade": g.grade.value,
                    "composite_score": g.composite_score,
                    "metrics": _metrics_dict(g.metrics),
                },
                "folds": [
                    {
                        "fold": f.fold,
                        "train_start": f.train_start,
                        "train_end": f.train_end,
                        "test_start": f.test_start,
                        "test_end": f.test_end,
                        "n_trades": f.n_trades,
                        "metrics": _metrics_dict(f.metrics),
                    }
                    for f in wf.folds
                ],
            })
            logger.info("    trades=%d  sharpe=%.3f  prec_buy=%.3f  grade=%s",
                        total_trades, wf.mean_sharpe, wf.mean_precision_buy, g.grade.value)
        except Exception as exc:
            logger.error("  FAILED %s: %s", name, exc)


def step_signals(
    exclude: list[str] | None = None,
    market: str = "us",
    feature_dir: Path = PARQUET_DIR,
    cache_dir: Path = CACHE_DIR,
) -> None:
    logger.info("[4/4] live signals — computing fresh from all strategies (no cache read)")
    # Fit on the labeled/trimmed training data — unchanged, since fit() needs
    # a real label and this keeps training/backtesting behavior untouched.
    df = load_training_data(feature_dir)

    # Fit every stateful strategy once, up front, on the full historical df —
    # fitting is a per-strategy cost, not a per-ticker one. Doing this before
    # touching any live feature data lets `df` be dropped before the ticker
    # loop below, instead of staying resident for the whole signals step.
    exclude_set = set(exclude or [])
    fitted_strategies: dict[str, "Strategy"] = {}
    for name in list_strategies():
        if name in exclude_set:
            logger.info("  signals: %s — skipped", name)
            continue
        try:
            strategy = load_strategy(name)
            if not _is_stateless(strategy):
                train_pd = _select_cols(df, strategy, OHLCV_COLS, FEATURE_COLS).to_pandas()
                if not getattr(strategy, "handles_nan", False):
                    train_pd = train_pd.dropna()
                if len(train_pd) < 100:
                    logger.warning("  %s: too few rows after dropna, skipping", name)
                    continue
                strategy.fit(train_pd)
            fitted_strategies[name] = strategy
        except Exception as exc:
            logger.error("  signals FAILED to fit %s: %s", name, exc)
    del df

    # Stream live features one ticker at a time (iter_live_features), instead
    # of materializing the whole universe plus a per-ticker-partitioned copy
    # of it. At 500-ticker scale (China CSI 500) that combination was the
    # reproducible cause of a Polars allocator crash
    # (`memory allocation of 15803824 bytes failed`) — see the Deviations
    # section of docs/superpowers/plans/2026-07-31-china-full-universe-training.md.
    # Peak memory here is bounded by a single ticker's feature frame.
    all_signals: list[dict] = []
    n_tickers = 0
    for ticker, t_pl in iter_live_features(market):
        n_tickers += 1
        if n_tickers % 50 == 0:
            logger.info("  signals: processed %d tickers so far", n_tickers)
        for name, strategy in fitted_strategies.items():
            t_pd = _select_cols(t_pl, strategy, OHLCV_COLS, FEATURE_COLS).to_pandas()
            if len(t_pd) == 0 or "time" not in t_pd.columns:
                continue
            try:
                result = strategy.predict(t_pd)
            except Exception as exc:
                logger.debug("  %s %s predict failed: %s", name, ticker, exc)
                continue

            # Last row = this ticker's own latest available trading day.
            pos = len(t_pd) - 1
            if pos >= len(result.signal):
                continue

            sig = str(result.signal.iloc[pos])
            conf = float(result.confidence.iloc[pos])
            close = float(t_pd["close"].iloc[pos]) if "close" in t_pd.columns else 0.0
            ticker_date = t_pd["time"].iloc[pos]
            trade = _trade_window(result.signal, t_pd["time"], pos)
            all_signals.append({
                "ticker": ticker,
                "date": str(ticker_date),
                "signal": sig,
                "confidence": conf,
                "entry_price": close,
                "position_size": conf,
                "strategy": name,
                "trade_status": trade["status"],
                "trade_open_date": trade["open_date"],
                "trade_close_date": trade["close_date"],
            })
    logger.info("  signals: done, %d tickers processed", n_tickers)

    buy_count = sum(1 for s in all_signals if s["signal"] == "Buy")
    logger.info("  total signals: %d  buy: %d", len(all_signals), buy_count)
    _write(cache_dir / "signals.json", {
        "generated_at": _now(),
        "signals": all_signals,
    })


# ---------------------------------------------------------------------------

def _pipeline_steps(signals_only: bool = False) -> list[str]:
    if signals_only:
        return ["data_summary", "signals"]
    return ["data_summary", "backtests", "leaderboard", "signals"]


def main(market: str = "us", signals_only: bool = False) -> None:
    feature_dir, cache_dir = _market_paths(market)
    steps = _pipeline_steps(signals_only)
    logger.info("=== Precomputing dashboard cache (%s) → %s  steps=%s ===", market, cache_dir, steps)
    if "data_summary" in steps:
        step_data_summary(feature_dir, cache_dir)
    if "backtests" in steps:
        step_backtests(feature_dir=feature_dir, cache_dir=cache_dir)
    if "leaderboard" in steps:
        step_leaderboard(cache_dir)
    if "signals" in steps:
        step_signals(market=market, feature_dir=feature_dir, cache_dir=cache_dir)
    logger.info("=== Done. Run: git add %s/ && git commit && git push ===", cache_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="us", choices=sorted(MARKETS))
    parser.add_argument(
        "--signals-only", action="store_true",
        help="Skip the expensive full walk-forward backtests/leaderboard steps; "
             "only refresh data summary + live signals (for frequent/daily runs "
             "against models kept current via scripts/incremental_train.py).",
    )
    args = parser.parse_args()
    main(args.market, signals_only=args.signals_only)
