"""Backtest the 3 newly-added diverse strategies (kalman_trend,
isolation_forest_anomaly, stacking_ensemble) at step_days=21 (monthly),
then rebuild leaderboard + signals so the dashboard reflects the trimmed +
expanded strategy set in one pass.
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

_NEW = ["kalman_trend", "isolation_forest_anomaly", "stacking_ensemble"]


def main() -> None:
    logger.info("=== Backtesting %d new strategies (step_days=21) ===", len(_NEW))
    step_backtests(_NEW, step_days=21)
    step_leaderboard()
    step_signals(exclude=["knn_classifier"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
