# src/strategies/rule_based/atr_breakout.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class ATRBreakout(Strategy):
    """Volatility breakout: buy when today's close exceeds the prior N-day high plus a fraction of ATR."""
    data_source = "ohlcv"

    def __init__(self, lookback: int = 10, atr_period: int = 14,
                 atr_multiplier: float = 0.5) -> None:
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        tr    = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"]  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr   = tr.ewm(span=self.atr_period, adjust=False).mean()

        prior_high = df["high"].shift(1).rolling(self.lookback).max()
        threshold  = prior_high + self.atr_multiplier * atr
        buy = close > threshold

        gap = ((close - threshold) / atr.replace(0, 1e-9)).clip(0, 1).fillna(0.0)
        confidence = gap.where(buy, 0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
