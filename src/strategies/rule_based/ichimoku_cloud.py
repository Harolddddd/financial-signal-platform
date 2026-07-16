from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class IchimokuCloud(Strategy):
    """Buy when price is above the cloud (Senkou A & B) AND Tenkan > Kijun."""
    data_source = "ohlcv"

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high = df["high"]
        low  = df["low"]
        close = df["close"]

        tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
        kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
        # Cloud lines: shift(26) → today's cloud was computed 26 bars ago
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        cloud_top = senkou_a.where(senkou_a >= senkou_b, senkou_b)

        above_cloud = close > cloud_top
        tenkan_above_kijun = tenkan > kijun
        buy = above_cloud & tenkan_above_kijun

        # Confidence: normalised gap between close and cloud top (capped at 5%)
        gap = ((close - cloud_top) / cloud_top.replace(0, 1e-9)).clip(0, 0.05) / 0.05
        confidence = gap.where(buy, 0.0).fillna(0.0)

        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
