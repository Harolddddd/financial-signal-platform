# src/strategies/rule_based/trend_pullback.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class TrendPullback(Strategy):
    """Buy-the-dip in an uptrend: triple MA bullish alignment + RSI cools to 35–50 zone."""
    data_source = "ohlcv"

    def __init__(self, short: int = 10, mid: int = 50, long: int = 200,
                 rsi_period: int = 14, rsi_low: float = 35.0,
                 rsi_high: float = 50.0) -> None:
        self.short = short
        self.mid = mid
        self.long = long
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        sma_s = close.rolling(self.short).mean()
        sma_m = close.rolling(self.mid).mean()
        sma_l = close.rolling(self.long).mean()
        uptrend = (sma_s > sma_m) & (sma_m > sma_l)

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

        pullback = (rsi >= self.rsi_low) & (rsi <= self.rsi_high)
        buy = uptrend & pullback

        # Confidence: midpoint distance within RSI window + trend strength
        rsi_mid  = (self.rsi_low + self.rsi_high) / 2
        rsi_conf = (1 - (rsi - rsi_mid).abs() / (rsi_high := self.rsi_high - rsi_mid)).clip(0, 1).fillna(0.0)
        gap1 = ((sma_s - sma_m) / sma_m.replace(0, 1e-9)).clip(0, 0.05) / 0.05
        trend_conf = gap1.fillna(0.0)
        confidence = ((rsi_conf + trend_conf) / 2).where(buy, 0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
