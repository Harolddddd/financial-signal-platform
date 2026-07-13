# src/strategies/rule_based/elder_ray.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class ElderRay(Strategy):
    """Elder Ray index: buy when Bull Power > 0 and Bear Power is negative but rising."""
    data_source = "ohlcv"

    def __init__(self, period: int = 13) -> None:
        self.period = period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        ema = df["close"].ewm(span=self.period, adjust=False).mean()
        bull_power = df["high"] - ema
        bear_power = df["low"]  - ema

        bear_rising = bear_power > bear_power.shift(1)
        buy = (bull_power > 0) & (bear_power < 0) & bear_rising

        # Confidence: normalised bull power magnitude
        bp_std = bull_power.rolling(50, min_periods=1).std().replace(0, 1e-9)
        confidence = (bull_power / bp_std).clip(0, 2).div(2).where(buy, 0.0).fillna(0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
