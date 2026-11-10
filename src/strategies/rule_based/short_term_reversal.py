# src/strategies/rule_based/short_term_reversal.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import PredictionResult, Strategy


class ShortTermReversal(Strategy):
    """Buy a sharp multi-day drop inside a long-term uptrend.

    Short-term (1-week) reversal is one of the most robust out-of-sample return
    patterns, strongest in retail-heavy markets such as China A-shares. The drop is
    measured in units of the stock's own daily volatility (a -5% week means more for
    a utility than for a chip maker); the 200-day trend filter skips falling knives.
    """

    data_source = "ohlcv"

    def __init__(self, lookback: int = 5, vol_window: int = 60, z_threshold: float = -2.0,
                 trend_window: int = 200) -> None:
        self.lookback = lookback
        self.vol_window = vol_window
        self.z_threshold = z_threshold
        self.trend_window = trend_window

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"].reset_index(drop=True)
        daily = close.pct_change()
        vol = daily.rolling(self.vol_window).std().replace(0, np.nan)
        z = close.pct_change(self.lookback) / (vol * np.sqrt(self.lookback))
        uptrend = close > close.rolling(self.trend_window).mean()
        buy = (z < self.z_threshold) & uptrend
        # z at the threshold -> 0.6 (the Buy line); twice the threshold -> 1.0
        confidence = (0.6 + 0.4 * (z / self.z_threshold - 1)).clip(0, 1).where(buy, 0.0).fillna(0.0)
        signal = pd.Series(np.where(buy.fillna(False), "Buy", "Hold"))
        return PredictionResult(confidence=confidence, signal=signal)
