# src/strategies/rule_based/triple_ma.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class TripleMAFilter(Strategy):
    """Buy only when short MA > mid MA > long MA (all three aligned bullish)."""
    data_source = "ohlcv"

    def __init__(self, short: int = 10, mid: int = 50, long: int = 200) -> None:
        self.short = short
        self.mid = mid
        self.long = long

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        sma_s = close.rolling(self.short).mean()
        sma_m = close.rolling(self.mid).mean()
        sma_l = close.rolling(self.long).mean()

        buy = (sma_s > sma_m) & (sma_m > sma_l)
        # Confidence: average of two gap ratios
        gap1 = ((sma_s - sma_m) / sma_m.replace(0, 1e-9)).clip(0, 0.05) / 0.05
        gap2 = ((sma_m - sma_l) / sma_l.replace(0, 1e-9)).clip(0, 0.05) / 0.05
        confidence = ((gap1 + gap2) / 2).fillna(0.0)
        confidence = confidence.where(buy, 0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
