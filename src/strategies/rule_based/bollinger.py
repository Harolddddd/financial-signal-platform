# src/strategies/rule_based/bollinger.py
from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class BollingerBounce(Strategy):
    data_source = "ohlcv"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = window
        self.num_std = num_std

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"]
        sma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper = sma + self.num_std * std
        lower = sma - self.num_std * std
        band_width = (upper - lower).replace(0, 1e-9)

        # %B: 0 = at lower band, 1 = at upper band; <0.2 = buy zone
        percent_b = (close - lower) / band_width
        # Confidence: how far into buy zone (0.2 threshold), capped at 1.0
        buy_threshold = 0.2
        confidence = ((buy_threshold - percent_b) / buy_threshold).clip(0, 1).fillna(0.0)
        signal = pd.Series(
            ["Buy" if v < buy_threshold else "Hold" for v in percent_b.fillna(1.0)],
            index=df.index,
        )
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
