# src/strategies/rule_based/momentum.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MomentumStrategy(Strategy):
    """Buy when N-day rate of change exceeds a positive threshold."""
    data_source = "ohlcv"

    def __init__(self, period: int = 20, threshold: float = 0.03) -> None:
        self.period = period
        self.threshold = threshold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        roc = close.pct_change(self.period)

        buy = roc > self.threshold
        # Confidence: how far above threshold, capped at 3× threshold
        confidence = ((roc - self.threshold) / (3 * self.threshold)).clip(0, 1).fillna(0.0)
        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
