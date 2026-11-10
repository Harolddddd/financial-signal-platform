# src/strategies/rule_based/high52_momentum.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import PredictionResult, Strategy


class High52Momentum(Strategy):
    """Buy stocks trading near their 52-week high (George & Hwang 2004).

    Investors anchor on the 52-week high and under-react to news that pushes the
    price toward it, so stocks close to the high keep drifting up. The effect is
    weak after bear markets and in high-volatility spells, so no Buy is given while
    the stock's own 21-day volatility is in the top fifth of its past year.
    """

    data_source = "ohlcv"

    def __init__(self, window: int = 252, near: float = 0.95, vol_window: int = 21,
                 vol_cap_quantile: float = 0.8) -> None:
        self.window = window
        self.near = near
        self.vol_window = vol_window
        self.vol_cap_quantile = vol_cap_quantile

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"].reset_index(drop=True)
        high = (df["high"] if "high" in df.columns else df["close"]).reset_index(drop=True)
        ratio = close / high.rolling(self.window, min_periods=self.window).max()
        vol = close.pct_change().rolling(self.vol_window).std()
        calm = vol <= vol.rolling(self.window, min_periods=self.window // 2).quantile(self.vol_cap_quantile)
        rising = close.pct_change(21) > 0
        buy = (ratio >= self.near) & calm & rising
        # ratio at `near` -> 0.6, at the high itself -> 1.0
        confidence = (0.6 + 0.4 * (ratio - self.near) / (1 - self.near)).clip(0, 1).where(buy, 0.0).fillna(0.0)
        signal = pd.Series(np.where(buy.fillna(False), "Buy", "Hold"))
        return PredictionResult(confidence=confidence, signal=signal)
