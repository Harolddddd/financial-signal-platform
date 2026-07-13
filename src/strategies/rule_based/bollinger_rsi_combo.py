# src/strategies/rule_based/bollinger_rsi_combo.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class BollingerRSICombo(Strategy):
    """Buy when price is near/below lower Bollinger Band AND RSI confirms not yet overbought."""
    data_source = "ohlcv"

    def __init__(self, bb_window: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14, rsi_max: float = 45.0) -> None:
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_max = rsi_max

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]

        sma = close.rolling(self.bb_window).mean()
        std = close.rolling(self.bb_window).std()
        lower = sma - self.bb_std * std
        band_width = (self.bb_std * 2 * std).replace(0, 1e-9)
        percent_b = (close - lower) / band_width

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

        bb_cond  = percent_b < 0.25
        rsi_cond = rsi < self.rsi_max
        buy = bb_cond & rsi_cond

        bb_conf  = ((0.25 - percent_b) / 0.25).clip(0, 1).fillna(0.0)
        rsi_conf = ((self.rsi_max - rsi) / self.rsi_max).clip(0, 1).fillna(0.0)
        confidence = ((bb_conf + rsi_conf) / 2).where(buy, 0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in buy.fillna(False)], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
