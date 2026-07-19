"""
Re-run all strategies on the full expanded dataset (no skip-if-exists).
Use this after expanding the ticker universe.

Usage:
    $env:PYTHONPATH="c:/Users/h1810/.vscode/EXP"
    python scripts/precompute_full.py
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dashboard.config import FEATURE_COLS, OHLCV_COLS, PARQUET_DIR
from src.backtesting.grader import grade_model
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.features.duckdb_client import load_training_data
from src.strategies.registry import list_strategies, load_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


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


def main() -> None:
    all_strategies = list_strategies()
    logger.info("=== Full precompute: %d strategies, expanded ticker universe ===", len(all_strategies))
    df = load_training_data(PARQUET_DIR)
    logger.info("Loaded %d rows across %d tickers", len(df), df["ticker"].n_unique())

    logger.info("[1/3] backtests — running all %d strategies (no skip)", len(all_strategies))
    for name in all_strategies:
        cache_path = CACHE_DIR / f"backtest_{_safe(name)}.json"
        logger.info("  strategy: %s", name)
        try:
            strategy = load_strategy(name)
            wf = walk_forward_backtest_strategy(
                df, strategy, OHLCV_COLS, FEATURE_COLS,
                train_window_days=400, test_window_days=21, step_days=21,
            )
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
            _write(cache_path, {
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
            logger.info(
                "    trades=%d  sharpe=%.3f  prec_buy=%.3f  grade=%s",
                total_trades, wf.mean_sharpe, wf.mean_precision_buy, g.grade.value,
            )
        except Exception as exc:
            logger.error("  FAILED %s: %s", name, exc)

    from scripts.precompute_dashboard import step_leaderboard, step_signals
    logger.info("[2/3] leaderboard — aggregating all %d backtest files", len(all_strategies))
    step_leaderboard()
    logger.info("[3/3] live signals — generating for all %d strategies", len(all_strategies))
    step_signals()

    logger.info("=== Done. Run: git add data/cache/ && git commit && git push ===")


if __name__ == "__main__":
    main()
