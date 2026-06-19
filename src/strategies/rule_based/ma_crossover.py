# src/strategies/rule_based/ma_crossover.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MACrossover(Strategy):
    data_source = "ohlcv"

    def __init__(self, fast_window: int = 20, slow_window: int = 50) -> None:
        self.fast_window = fast_window
        self.slow_window = slow_window

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        fast = close.rolling(self.fast_window).mean()
        slow = close.rolling(self.slow_window).mean()

        raw_diff = (fast - slow) / slow.abs().clip(lower=1e-9)
        # Confidence: normalize gap to [0, 1]; 5% gap above slow = full confidence
        confidence = raw_diff.clip(0, 0.05).div(0.05).fillna(0.0)
        signal = pd.Series(
            ["Buy" if d > 0 else "Hold" for d in raw_diff.fillna(0.0)],
            index=df.index,
        )
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
