# src/strategies/rule_based/stochastic.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class StochasticOscillator(Strategy):
    data_source = "ohlcv"

    def __init__(self, k_period: int = 14, d_period: int = 3, oversold: float = 20.0) -> None:
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        low_min  = df["low"].rolling(self.k_period).min()
        high_max = df["high"].rolling(self.k_period).max()
        denom = (high_max - low_min).replace(0, 1e-9)
        pct_k = 100 * (df["close"] - low_min) / denom
        pct_d = pct_k.rolling(self.d_period).mean()

        # Buy when %K crosses above %D in oversold territory
        in_oversold = pct_k < self.oversold
        k_above_d   = pct_k > pct_d
        buy = in_oversold & k_above_d

        confidence = ((self.oversold - pct_k.clip(upper=self.oversold)) / self.oversold).clip(0, 1).fillna(0.0)
        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
