"""Rebuild signals.json after dropping stacking_ensemble (D grade, negative
Sharpe, ~40x the compute cost of any other strategy — not worth keeping).
knn_classifier stays excluded — its live-signal fit/predict takes hours.
"""
import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.utils\.parallel")

from scripts.precompute_dashboard import step_signals

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Rebuilding signals.json (stacking_ensemble dropped) ===")
    step_signals(exclude=["knn_classifier"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
