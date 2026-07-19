import numpy as np
import pandas as pd

from src.strategies.base import Strategy, PredictionResult


class TRIX(Strategy):
    data_source = "ohlcv"

    def __init__(self, period: int = 15) -> None:
        self.period = period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]

        ema1 = close.ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()

        prev_ema3 = ema3.shift(1)
        trix = ((ema3 / prev_ema3.replace(0.0, float("nan"))) - 1) * 100

        buy = trix > 0
        confidence = (trix / 2.0).clip(0, 1)

        signal = pd.Series(np.where(buy, "Buy", "Hold"), index=df.index)
        conf_out = pd.Series(np.where(buy, confidence, 0.0), index=df.index)

        return PredictionResult(confidence=conf_out, signal=signal)
