# src/strategies/rule_based/williams_r.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class WilliamsR(Strategy):
    data_source = "ohlcv"

    def __init__(self, period: int = 14, oversold: float = -80.0) -> None:
        self.period = period
        self.oversold = oversold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high_max = df["high"].rolling(self.period).max()
        low_min  = df["low"].rolling(self.period).min()
        denom = (high_max - low_min).replace(0, 1e-9)
        wr = -100 * (high_max - df["close"]) / denom

        buy = wr < self.oversold
        # Confidence: 0 at the oversold threshold, 1 at maximum oversold (-100).
        # Range = 100 + oversold (e.g. 20 when oversold=-80).
        confidence = ((self.oversold - wr) / (100.0 + self.oversold)).clip(0, 1).fillna(0.0)
        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
