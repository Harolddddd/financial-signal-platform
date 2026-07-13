# src/strategies/rule_based/ma_volume_confirm.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class MAVolumeConfirm(Strategy):
    """EMA uptrend (fast > slow) confirmed by a volume surge above its rolling average."""
    data_source = "ohlcv"

    def __init__(self, fast: int = 20, slow: int = 50,
                 vol_window: int = 20, vol_multiplier: float = 1.5) -> None:
        self.fast = fast
        self.slow = slow
        self.vol_window = vol_window
        self.vol_multiplier = vol_multiplier

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        trend_up = ema_fast > ema_slow

        avg_vol  = df["volume"].rolling(self.vol_window).mean().replace(0, 1e-9)
        vol_surge = df["volume"] > self.vol_multiplier * avg_vol
        buy = trend_up & vol_surge

        trend_conf = ((ema_fast - ema_slow) / ema_slow.replace(0, 1e-9)).clip(0, 0.05) / 0.05
        vol_conf   = ((df["volume"] / avg_vol - self.vol_multiplier) / self.vol_multiplier).clip(0, 1).fillna(0.0)
        confidence = ((trend_conf + vol_conf) / 2).where(buy, 0.0).fillna(0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
