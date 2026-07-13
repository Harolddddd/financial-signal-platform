# src/strategies/rule_based/rsi_macd_combo.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class RSIMACDCombo(Strategy):
    """Buy when RSI is oversold AND MACD histogram turns positive — dual confirmation."""
    data_source = "ohlcv"

    def __init__(self, rsi_period: int = 14, rsi_oversold: float = 35.0,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> None:
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line

        rsi_cond  = rsi < self.rsi_oversold
        macd_cond = histogram > 0
        buy = rsi_cond & macd_cond

        rsi_conf  = ((self.rsi_oversold - rsi) / self.rsi_oversold).clip(0, 1).fillna(0.0)
        roll_max  = histogram.clip(lower=0).rolling(50, min_periods=1).max().replace(0, 1e-9)
        macd_conf = (histogram.clip(lower=0) / roll_max).fillna(0.0)
        confidence = ((rsi_conf + macd_conf) / 2).where(buy, 0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
