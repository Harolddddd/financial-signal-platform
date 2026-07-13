# src/strategies/rule_based/parabolic_sar.py
from __future__ import annotations
import pandas as pd
import numpy as np
from src.strategies.base import Strategy, PredictionResult


class ParabolicSAR(Strategy):
    """Buy on Parabolic SAR flip: price rises above the SAR trailing stop."""
    data_source = "ohlcv"

    def __init__(self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> None:
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        high  = df["high"].to_numpy()
        low   = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        sar    = np.zeros(n)
        trend  = np.zeros(n, dtype=int)   # 1 = bull, -1 = bear
        ep     = np.zeros(n)
        af     = np.zeros(n)

        # Initialise
        sar[0]   = low[0]
        trend[0] = 1
        ep[0]    = high[0]
        af[0]    = self.af_start

        for i in range(1, n):
            prev_sar = sar[i - 1]
            prev_ep  = ep[i - 1]
            prev_af  = af[i - 1]
            prev_tr  = trend[i - 1]

            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            if prev_tr == 1:
                new_sar = min(new_sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
                if low[i] < new_sar:
                    trend[i] = -1
                    new_sar  = prev_ep
                    new_ep   = low[i]
                    new_af   = self.af_start
                else:
                    trend[i] = 1
                    new_ep   = max(prev_ep, high[i])
                    new_af   = min(prev_af + self.af_step, self.af_max) if new_ep > prev_ep else prev_af
            else:
                new_sar = max(new_sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
                if high[i] > new_sar:
                    trend[i] = 1
                    new_sar  = prev_ep
                    new_ep   = high[i]
                    new_af   = self.af_start
                else:
                    trend[i] = -1
                    new_ep   = min(prev_ep, low[i])
                    new_af   = min(prev_af + self.af_step, self.af_max) if new_ep < prev_ep else prev_af

            sar[i] = new_sar
            ep[i]  = new_ep
            af[i]  = new_af

        bull = pd.Series(trend == 1, index=df.index)
        # Confidence: distance of close above SAR, normalised by ATR
        atr = (df["high"] - df["low"]).rolling(14).mean().replace(0, 1e-9)
        gap = pd.Series((close - sar) / atr.to_numpy(), index=df.index).clip(0, 2) / 2
        confidence = gap.where(bull, 0.0).fillna(0.0)

        signal = pd.Series(["Buy" if b else "Hold" for b in bull], index=df.index)
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                signal=signal.reset_index(drop=True))
