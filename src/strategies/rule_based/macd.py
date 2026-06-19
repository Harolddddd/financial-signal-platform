# src/strategies/rule_based/macd.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MACDSignal(Strategy):
    data_source = "ohlcv"

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        # Confidence: positive histogram normalized to rolling 50-bar max
        pos_hist = histogram.clip(lower=0)
        roll_max = pos_hist.rolling(50, min_periods=1).max().replace(0, 1e-9)
        confidence = (pos_hist / roll_max).fillna(0.0)
        signal = pd.Series(
            ["Buy" if h > 0 else "Hold" for h in histogram.fillna(0.0)],
            index=df.index,
        )
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
