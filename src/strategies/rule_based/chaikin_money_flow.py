from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class ChaikinMoneyFlow(Strategy):
    """Buy when the 20-period Chaikin Money Flow is positive (> 0)."""
    data_source = "ohlcv"

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        vol   = df["volume"]

        hl_range = (high - low).replace(0.0, float("nan"))
        mfm = ((close - low) - (high - close)) / hl_range  # money flow multiplier [-1, 1]
        mfv = mfm * vol                                      # money flow volume

        cmf = (
            mfv.rolling(self.period).sum()
            / vol.rolling(self.period).sum().replace(0.0, float("nan"))
        )

        buy = cmf > 0.0
        confidence = cmf.clip(0.0, 1.0).where(buy, 0.0).fillna(0.0)
        signal = pd.Series(
            ["Buy" if b else "Hold" for b in buy.fillna(False)],
            index=df.index,
        )
        return PredictionResult(
            confidence=confidence.reset_index(drop=True),
            signal=signal.reset_index(drop=True),
        )
