"""Resume scripts/precompute_dashboard.py after it was killed partway through
step_backtests() — reruns only the strategies that never got a completed
backtest cache, then regenerates the leaderboard and signals caches on top.
"""
import logging
import sys
import warnings

# ExtraTreesClassifier (and possibly others) re-emit a UserWarning on every
# joblib parallel call instead of once per process — across ~780 folds x 492
# tickers that's hundreds of thousands of lines to stderr. filterwarnings()
# and PYTHONWARNINGS both failed to stop it at the source (loky worker /
# re-emission internals), so instead: real progress logs go to stdout
# (explicitly, since precompute_dashboard's own basicConfig() already
# grabbed the root logger's default stderr handler on import and a plain
# basicConfig() call here would be a no-op), and stderr is discarded
# entirely by the caller (`2>$null` / `2>/dev/null`) so the noise never
# hits disk regardless of which layer is generating it. Default joblib
# backend (loky, separate processes) stays on for full multi-core speed —
# forcing "threading" earlier only cut effective parallelism (GIL-bound)
# without being needed for the flood fix, which is stream-level not
# backend-level.
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.utils\.parallel")

from scripts.precompute_dashboard import step_backtests, step_leaderboard, step_signals

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
logger = logging.getLogger(__name__)

_REMAINING = [
    "adaboost", "adx_trend", "aroon_oscillator", "catboost_strategy",
    "chaikin_money_flow", "gaussian_nb", "hist_gradient_boosting", "hull_ma",
    "ichimoku_cloud", "knn_classifier", "lda_strategy", "mlp_classifier",
    "money_flow_index", "stoch_rsi", "supertrend", "svm_strategy", "trix", "vwap_cross",
]
# Wider walk-forward step for the remaining strategies — quarterly folds
# instead of monthly, ~3x fewer folds, still covers the full history span.
_STEP_DAYS = 63


def main() -> None:
    logger.info("=== Resuming precompute for %d remaining strategies (step_days=%d) ===", len(_REMAINING), _STEP_DAYS)
    step_backtests(_REMAINING, step_days=_STEP_DAYS)
    step_leaderboard()
    step_signals()
    logger.info("=== Resume complete ===")


if __name__ == "__main__":
    main()
