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

from dashboard.config import CONFIDENCE_THRESHOLD, FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
from dashboard.data_loader import (
    get_backtest_result,
    get_data_summary,
    get_leaderboard,
    get_live_signals,
)
from src.strategies.registry import list_strategies

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
    summary = get_data_summary(PARQUET_DIR)
    summary["generated_at"] = _now()
    _write(CACHE_DIR / "data_summary.json", summary)


def step_leaderboard() -> None:
    logger.info("[2/4] leaderboard (walk-forward for all strategies)")
    grades = get_leaderboard(PARQUET_DIR, OHLCV_COLS, FEATURE_COLS)
    _write(CACHE_DIR / "leaderboard.json", {
        "generated_at": _now(),
        "grades": [
            {
                "model_name": g.model_name,
                "grade": g.grade.value,
                "composite_score": g.composite_score,
                "metrics": _metrics_dict(g.metrics),
            }
            for g in grades
        ],
    })


def step_backtests() -> None:
    logger.info("[3/4] per-strategy backtests")
    for name in list_strategies():
        logger.info("  strategy: %s", name)
        try:
            wf, grade = get_backtest_result(name, PARQUET_DIR, OHLCV_COLS, FEATURE_COLS)
            _write(CACHE_DIR / f"backtest_{_safe(name)}.json", {
                "generated_at": _now(),
                "strategy_name": name,
                "mean_sharpe": wf.mean_sharpe,
                "mean_win_rate": wf.mean_win_rate,
                "mean_precision_buy": wf.mean_precision_buy,
                "worst_drawdown": wf.worst_drawdown,
                "grade": {
                    "model_name": grade.model_name,
                    "grade": grade.grade.value,
                    "composite_score": grade.composite_score,
                    "metrics": _metrics_dict(grade.metrics),
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
        except Exception as exc:
            logger.error("  FAILED %s: %s", name, exc)


def step_signals() -> None:
    logger.info("[4/4] live signals (threshold=0.0 — store all, filter at display time)")
    # Store at threshold=0.0 so the dashboard slider can re-filter without recomputing.
    signals = get_live_signals(PARQUET_DIR, OHLCV_COLS, FEATURE_COLS, confidence_threshold=0.0)
    _write(CACHE_DIR / "signals.json", {
        "generated_at": _now(),
        "signals": [
            {
                "ticker": s.ticker,
                "date": s.date,
                "signal": s.signal.value,
                "confidence": s.confidence,
                "entry_price": s.entry_price,
                "position_size": s.position_size,
            }
            for s in signals
        ],
    })


# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Precomputing dashboard cache → %s ===", CACHE_DIR)
    step_data_summary()
    step_leaderboard()
    step_backtests()
    step_signals()
    logger.info("=== Done. Run: git add data/cache/ && git commit && git push ===")


if __name__ == "__main__":
    main()
