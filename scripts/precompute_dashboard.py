"""
Run locally to pre-compute all dashboard data and write JSON to data/cache/.
Commit data/cache/ and push — Render reads from these files instead of
recomputing on every cold start.

Usage:
    python scripts/precompute_dashboard.py
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from dashboard.config import CONFIDENCE_THRESHOLD, FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
from src.backtesting.grader import grade_model
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.strategy_runner import _is_stateless, _select_cols, walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.strategies.registry import list_strategies, load_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


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


# ---------------------------------------------------------------------------

def step_data_summary() -> None:
    logger.info("[1/4] data summary")
    # Read directly from parquets — never read from cache here, that defeats the purpose.
    df = load_training_data(PARQUET_DIR)
    tickers = sorted(df["ticker"].unique().to_list()) if "ticker" in df.columns else []
    summary = {
        "n_tickers": len(tickers),
        "n_rows": len(df),
        "tickers": tickers,
        "date_range_start": str(df["time"].min()) if "time" in df.columns else "N/A",
        "date_range_end":   str(df["time"].max()) if "time" in df.columns else "N/A",
        "generated_at": _now(),
    }
    _write(CACHE_DIR / "data_summary.json", summary)


def step_leaderboard() -> None:
    logger.info("[3/4] leaderboard — aggregating from per-strategy backtest caches")
    # Build the leaderboard from the individual backtest files written in step_backtests().
    # Never call get_leaderboard() here — it reads the old leaderboard.json and writes it back.
    grades: list[dict] = []
    for name in list_strategies():
        cache_path = CACHE_DIR / f"backtest_{_safe(name)}.json"
        if cache_path.exists():
            d = json.loads(cache_path.read_text())
            grades.append(d["grade"])
        else:
            logger.warning("  no backtest cache for %s — skipping from leaderboard", name)

    grades.sort(key=lambda g: g["composite_score"], reverse=True)
    _write(CACHE_DIR / "leaderboard.json", {
        "generated_at": _now(),
        "grades": grades,
    })


def step_backtests() -> None:
    logger.info("[2/4] per-strategy backtests — running walk-forward fresh (no cache read)")
    # Load data once; reuse across all strategies.
    df = load_training_data(PARQUET_DIR)
    for name in list_strategies():
        logger.info("  strategy: %s", name)
        try:
            strategy = load_strategy(name)
            wf = walk_forward_backtest_strategy(
                df, strategy, OHLCV_COLS, FEATURE_COLS,
                train_window_days=400, test_window_days=21, step_days=21,
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
            _write(CACHE_DIR / f"backtest_{_safe(name)}.json", {
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


def step_signals() -> None:
    logger.info("[4/4] live signals — computing fresh from all strategies (no cache read)")
    df = load_training_data(PARQUET_DIR)
    latest_date = df["time"].max()
    has_ticker = "ticker" in df.columns
    tickers: list[str] = sorted(df["ticker"].unique().to_list()) if has_ticker else ["__all__"]

    # Pre-partition by ticker once.
    ticker_pdfs: dict[str, dict] = {}
    for ticker in tickers:
        t_pl = df.filter(pl.col("ticker") == ticker) if has_ticker else df
        ticker_pdfs[ticker] = t_pl  # keep as Polars for per-strategy column selection

    all_signals: list[dict] = []
    for name in list_strategies():
        logger.info("  signals: %s", name)
        try:
            strategy = load_strategy(name)

            if not _is_stateless(strategy):
                # Fit on all data for a live signal (no train/test split needed).
                train_pd = _select_cols(df, strategy, OHLCV_COLS, FEATURE_COLS).to_pandas()
                if not getattr(strategy, "handles_nan", False):
                    train_pd = train_pd.dropna()
                if len(train_pd) < 100:
                    logger.warning("  %s: too few rows after dropna, skipping", name)
                    continue
                strategy.fit(train_pd)

            for ticker in tickers:
                t_pd = _select_cols(ticker_pdfs[ticker], strategy, OHLCV_COLS, FEATURE_COLS).to_pandas()
                if len(t_pd) == 0:
                    continue
                try:
                    result = strategy.predict(t_pd)
                except Exception as exc:
                    logger.debug("  %s %s predict failed: %s", name, ticker, exc)
                    continue

                # Find the row matching the latest date.
                if "time" not in t_pd.columns:
                    continue
                latest_rows = t_pd[t_pd["time"] == latest_date]
                if latest_rows.empty:
                    continue
                pos = t_pd.index.get_loc(latest_rows.index[0])
                if pos >= len(result.signal):
                    continue

                sig = str(result.signal.iloc[pos])
                conf = float(result.confidence.iloc[pos])
                close = float(t_pd["close"].iloc[pos]) if "close" in t_pd.columns else 0.0
                all_signals.append({
                    "ticker": ticker if ticker != "__all__" else "ALL",
                    "date": str(latest_date),
                    "signal": sig,
                    "confidence": conf,
                    "entry_price": close,
                    "position_size": conf,
                    "strategy": name,
                })
        except Exception as exc:
            logger.error("  signals FAILED %s: %s", name, exc)

    buy_count = sum(1 for s in all_signals if s["signal"] == "Buy")
    logger.info("  total signals: %d  buy: %d", len(all_signals), buy_count)
    _write(CACHE_DIR / "signals.json", {
        "generated_at": _now(),
        "signals": all_signals,
    })


# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Precomputing dashboard cache → %s ===", CACHE_DIR)
    step_data_summary()   # [1/4] parquet → data_summary.json
    step_backtests()      # [2/4] per-strategy walk-forward → backtest_*.json
    step_leaderboard()    # [3/4] aggregate backtest_*.json → leaderboard.json
    step_signals()        # [4/4] live signals → signals.json
    logger.info("=== Done. Run: git add data/cache/ && git commit && git push ===")


if __name__ == "__main__":
    main()
