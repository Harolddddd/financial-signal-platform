from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class VWAPCross(Strategy):
    """Buy when close is above the rolling VWAP (price-above-VWAP trend confirmation)."""
    data_source = "ohlcv"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        vol   = df["volume"]
        typical = (df["high"] + df["low"] + close) / 3

        vol_sum = vol.rolling(self.window).sum().replace(0.0, float("nan"))
        vwap = (typical * vol).rolling(self.window).sum() / vol_sum

        buy = close > vwap
        # Confidence: percentage gap above VWAP, scaled so 10% above → confidence=1
        gap = ((close - vwap) / vwap.replace(0.0, float("nan"))).clip(0.0, 0.10) / 0.10
        confidence = gap.where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
