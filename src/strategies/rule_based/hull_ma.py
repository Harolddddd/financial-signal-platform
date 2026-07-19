import numpy as np
import pandas as pd

from src.strategies.base import Strategy, PredictionResult


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


class HullMA(Strategy):
    data_source = "ohlcv"

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        half = max(self.period // 2, 1)
        sqrt_period = max(int(np.sqrt(self.period)), 1)

        wma_half = _wma(close, half)
        wma_full = _wma(close, self.period)
        raw = 2 * wma_half - wma_full
        hma = _wma(raw, sqrt_period)

        buy = close > hma
        gap = ((close - hma) / close).clip(0, 0.02) / 0.02

        signal = pd.Series(np.where(buy, "Buy", "Hold"), index=df.index)
        confidence = pd.Series(np.where(buy, gap, 0.0), index=df.index)

        return PredictionResult(confidence=confidence, signal=signal)
