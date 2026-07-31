"""Finish backtest_new_strategies.py after it was killed mid-stacking_ensemble
run — the sklearn.utils.parallel UserWarning flood (worse here than usual
because StackingClassifier nests nested Parallel() calls per base estimator
per CV fold) filled a 2.5GB stderr log in ~90 minutes. kalman_trend and
isolation_forest_anomaly already completed and are cached; this just redoes
stacking_ensemble, then rebuilds leaderboard + signals.
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
    logger.info("=== Backtesting stacking_ensemble only (step_days=21) ===")
    step_backtests(["stacking_ensemble"], step_days=21)
    step_leaderboard()
    step_signals(exclude=["knn_classifier"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
