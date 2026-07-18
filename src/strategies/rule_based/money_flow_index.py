from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MoneyFlowIndex(Strategy):
    """Buy when MFI is below the oversold threshold (volume-weighted RSI recovery signal)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 14, oversold: float = 30.0) -> None:
        self.period = period
        self.oversold = oversold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        prev_typical = typical.shift(1)
        raw_flow = typical * df["volume"]

        positive_flow = raw_flow.where(typical > prev_typical, 0.0)
        negative_flow = raw_flow.where(typical < prev_typical, 0.0)

        pos_sum = positive_flow.rolling(self.period).sum()
        neg_sum = negative_flow.rolling(self.period).sum().replace(0.0, float("nan"))

        mfr = pos_sum / neg_sum
        mfi = 100.0 - (100.0 / (1.0 + mfr))

        buy = mfi < self.oversold
        confidence = ((self.oversold - mfi) / self.oversold).clip(0.0, 1.0).where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
