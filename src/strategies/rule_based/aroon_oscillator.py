from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class AroonOscillator(Strategy):
    """Buy when the Aroon oscillator is above +50 (strong upward momentum)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 25) -> None:
        self.period = period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high = df["high"]
        low  = df["low"]
        p = self.period

        aroon_up   = (high.rolling(p + 1).apply(lambda x: x.argmax(), raw=True) / p) * 100
        aroon_down = (low.rolling(p + 1).apply(lambda x: x.argmin(), raw=True)  / p) * 100
        aroon_osc  = aroon_up - aroon_down  # range [-100, 100]

        buy = aroon_osc > 50.0
        # Confidence: scale oscillator from 50–100 → 0–1
        confidence = ((aroon_osc - 50.0) / 50.0).clip(0.0, 1.0).where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
