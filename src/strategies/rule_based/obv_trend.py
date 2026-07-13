# src/strategies/rule_based/obv_trend.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class OBVTrend(Strategy):
    """Buy when On-Balance Volume is trending up but price has pulled back."""
    data_source = "ohlcv"

    def __init__(self, obv_window: int = 20, price_window: int = 5) -> None:
        self.obv_window = obv_window
        self.price_window = price_window

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * df["volume"]).cumsum()

        obv_rising  = obv > obv.rolling(self.obv_window).mean()
        price_dip   = df["close"] < df["close"].rolling(self.price_window).mean()
        buy = obv_rising & price_dip

        # Confidence: OBV distance above its MA, normalized
        obv_ma  = obv.rolling(self.obv_window).mean()
        obv_std = obv.rolling(self.obv_window).std().replace(0, 1e-9)
        z = ((obv - obv_ma) / obv_std).clip(0, 2)
        confidence = (z / 2).fillna(0.0)
        confidence = confidence.where(buy, 0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
