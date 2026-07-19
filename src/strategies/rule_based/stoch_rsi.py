import numpy as np
import pandas as pd

from src.strategies.base import Strategy, PredictionResult


class StochRSI(Strategy):
    data_source = "ohlcv"

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14, oversold: float = 0.2) -> None:
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.oversold = oversold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, float("nan"))
        rsi = 100 - 100 / (1 + rs)

        rsi_min = rsi.rolling(self.stoch_period).min()
        rsi_max = rsi.rolling(self.stoch_period).max()
        denom = (rsi_max - rsi_min).replace(0.0, float("nan"))
        stoch_rsi = ((rsi - rsi_min) / denom).clip(0, 1)

        buy = stoch_rsi < self.oversold
        confidence = ((self.oversold - stoch_rsi) / self.oversold).clip(0, 1)

        signal = pd.Series(np.where(buy, "Buy", "Hold"), index=df.index)
        conf_out = pd.Series(np.where(buy, confidence, 0.0), index=df.index)

        return PredictionResult(confidence=conf_out, signal=signal)
