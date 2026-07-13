# src/strategies/rule_based/pvt_trend.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class PVTTrend(Strategy):
    """Price-Volume Trend: buy when PVT is rising (volume-weighted buying) but price has pulled back."""
    data_source = "ohlcv"

    def __init__(self, pvt_window: int = 20, price_window: int = 5) -> None:
        self.pvt_window = pvt_window
        self.price_window = price_window

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        pct_change = df["close"].pct_change().fillna(0.0)
        pvt = (pct_change * df["volume"]).cumsum()

        pvt_rising  = pvt > pvt.rolling(self.pvt_window).mean()
        price_dip   = df["close"] < df["close"].rolling(self.price_window).mean()
        buy = pvt_rising & price_dip

        pvt_ma  = pvt.rolling(self.pvt_window).mean()
        pvt_std = pvt.rolling(self.pvt_window).std().replace(0, 1e-9)
        z = ((pvt - pvt_ma) / pvt_std).clip(0, 2) / 2
        confidence = z.where(buy, 0.0).fillna(0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
