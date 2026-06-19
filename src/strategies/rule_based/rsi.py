# src/strategies/rule_based/rsi.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class RSIThreshold(Strategy):
    data_source = "ohlcv"

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))

        # BUY when RSI < oversold; confidence = distance below threshold / threshold
        confidence = ((self.oversold - rsi) / self.oversold).clip(0, 1).fillna(0.0)
        signal = pd.Series(
            ["Buy" if v < self.oversold else "Hold" for v in rsi.fillna(50)],
            index=df.index,
        )
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
