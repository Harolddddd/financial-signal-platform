from __future__ import annotations
import numpy as np
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class SuperTrend(Strategy):
    """Buy when close is above the ATR-based SuperTrend line (uptrend confirmation)."""
    data_source = "ohlcv"

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0) -> None:
        self.atr_period = atr_period
        self.multiplier = multiplier

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        hl2 = (high + low) / 2
        upper_band = (hl2 + self.multiplier * atr).to_numpy()
        lower_band = (hl2 - self.multiplier * atr).to_numpy()
        close_arr = close.to_numpy()

        # Iterative: each bar depends on the previous supertrend value
        n = len(df)
        st_arr = lower_band.copy()
        for i in range(1, n):
            prev = st_arr[i - 1]
            if np.isnan(prev) or np.isnan(lower_band[i]) or np.isnan(upper_band[i]):
                st_arr[i] = lower_band[i]
                continue
            if close_arr[i] > prev:
                st_arr[i] = max(lower_band[i], prev)
            else:
                st_arr[i] = min(upper_band[i], prev)

        st_series = pd.Series(st_arr, index=df.index)
        buy = close > st_series
        gap = ((close - st_series) / close.replace(0.0, float("nan"))).clip(0.0, 0.05) / 0.05
        confidence = gap.where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
