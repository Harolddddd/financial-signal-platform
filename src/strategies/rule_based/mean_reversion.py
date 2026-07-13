# src/strategies/rule_based/mean_reversion.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MeanReversion(Strategy):
    """Buy when price is statistically cheap vs. its rolling mean (z-score < threshold)."""
    data_source = "ohlcv"

    def __init__(self, window: int = 60, z_threshold: float = -1.5) -> None:
        self.window = window
        self.z_threshold = z_threshold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        ma  = close.rolling(self.window).mean()
        std = close.rolling(self.window).std().replace(0, 1e-9)
        z   = (close - ma) / std

        buy = z < self.z_threshold
        # Confidence: how far below threshold (more negative = stronger signal)
        confidence = ((self.z_threshold - z) / abs(self.z_threshold)).clip(0, 1).fillna(0.0)
        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
