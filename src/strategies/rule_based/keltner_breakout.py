# src/strategies/rule_based/keltner_breakout.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class KeltnerBreakout(Strategy):
    """Buy when price breaks above the Keltner Channel upper band (EMA + multiplier × ATR)."""
    data_source = "ohlcv"

    def __init__(self, ema_period: int = 20, atr_period: int = 14,
                 multiplier: float = 2.0) -> None:
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        ema   = close.ewm(span=self.ema_period, adjust=False).mean()
        tr    = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"]  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr   = tr.ewm(span=self.atr_period, adjust=False).mean()
        upper = ema + self.multiplier * atr

        buy = close > upper
        gap = ((close - upper) / atr.replace(0, 1e-9)).clip(0, 1).fillna(0.0)
        confidence = gap.where(buy, 0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
