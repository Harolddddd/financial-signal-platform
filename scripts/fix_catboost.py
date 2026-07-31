"""catboost was never installed, so catboost_strategy silently failed in both
step_backtests() and step_signals() during the precompute run, leaving a
stale pre-session cache. Now that catboost is installed, redo its backtest
(same quarterly step_days as the rest of this run's later batch) and
regenerate the leaderboard + signals on top so it's included consistently.
"""
import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.utils\.parallel")

from scripts.precompute_dashboard import step_backtests, step_leaderboard, step_signals

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Fixing catboost_strategy (step_days=63) ===")
    step_backtests(["catboost_strategy"], step_days=63)
    step_leaderboard()
    step_signals(exclude=["knn_classifier"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
