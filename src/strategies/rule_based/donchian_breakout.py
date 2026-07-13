# src/strategies/rule_based/donchian_breakout.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class DonchianBreakout(Strategy):
    """Buy when price breaks above the N-day high (momentum breakout)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        # Use prior period's high to avoid look-ahead
        prior_high = df["high"].shift(1).rolling(self.period).max()
        breakout = df["close"] > prior_high

        # Confidence: how far above the channel top
        gap = ((df["close"] - prior_high) / prior_high.replace(0, 1e-9)).clip(0, 0.05)
        confidence = (gap / 0.05).fillna(0.0)
        signal = pd.Series(["Buy" if b else "Hold" for b in breakout.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
