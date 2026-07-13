# src/strategies/rule_based/cci.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class CCIStrategy(Strategy):
    data_source = "ohlcv"

    def __init__(self, period: int = 20, oversold: float = -100.0) -> None:
        self.period = period
        self.oversold = oversold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        sma = typical.rolling(self.period).mean()
        mean_dev = typical.rolling(self.period).apply(lambda x: (abs(x - x.mean())).mean(), raw=True)
        cci = (typical - sma) / (0.015 * mean_dev.replace(0, 1e-9))

        # Buy when CCI rises from below oversold threshold
        buy = cci < self.oversold
        # Confidence: how far below threshold (deeper oversold = more confident)
        confidence = ((self.oversold - cci) / abs(self.oversold)).clip(0, 1).fillna(0.0)
        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
