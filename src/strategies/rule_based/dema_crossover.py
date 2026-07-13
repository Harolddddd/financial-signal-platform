# src/strategies/rule_based/dema_crossover.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class DEMACrossover(Strategy):
    """Double EMA (DEMA) fast/slow crossover — reacts faster than a simple SMA crossover."""
    data_source = "ohlcv"

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        self.fast = fast
        self.slow = slow

    @staticmethod
    def _dema(series: pd.Series, period: int) -> pd.Series:
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        dema_fast = self._dema(close, self.fast)
        dema_slow = self._dema(close, self.slow)

        buy = dema_fast > dema_slow
        gap = ((dema_fast - dema_slow) / dema_slow.replace(0, 1e-9)).clip(0, 0.05) / 0.05
        confidence = gap.where(buy, 0.0).fillna(0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
