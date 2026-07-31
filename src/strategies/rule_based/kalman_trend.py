# src/strategies/rule_based/kalman_trend.py
from __future__ import annotations
import numpy as np
import pandas as pd
from src.strategies.base import Strategy, PredictionResult


class KalmanTrend(Strategy):
    """Local-linear-trend Kalman filter on close price (state = [level, slope]).
    Buys when price is above the filtered level and the filtered slope is
    positive — an adaptive trend estimate, structurally different from the
    fixed-window moving averages used elsewhere in this strategy set."""
    data_source = "ohlcv"

    def __init__(self, process_var: float = 1e-4, obs_var: float = 1.0,
                 slope_threshold: float = 0.0) -> None:
        self.process_var = process_var
        self.obs_var = obs_var
        self.slope_threshold = slope_threshold

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        close = df["close"].to_numpy(dtype=float)
        n = len(close)

        # State = [level, slope]; transition: level_t = level_{t-1} + slope_{t-1}.
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * self.process_var
        R = self.obs_var

        x = np.array([close[0], 0.0])
        P = np.eye(2)

        levels = np.empty(n)
        slopes = np.empty(n)
        for t in range(n):
            if t > 0:
                x = F @ x
                P = F @ P @ F.T + Q
            innovation = close[t] - (H @ x)[0]
            S = (H @ P @ H.T)[0, 0] + R
            K = (P @ H.T).flatten() / S
            x = x + K * innovation
            P = (np.eye(2) - np.outer(K, H)) @ P
            levels[t] = x[0]
            slopes[t] = x[1]

        buy = (close > levels) & (slopes > self.slope_threshold)
        norm_slope = np.clip(slopes / (np.abs(close) + 1e-9) * 100.0, 0.0, 1.0)
        confidence = pd.Series(np.where(buy, norm_slope, 0.0))
        signal = pd.Series(["Buy" if b else "Hold" for b in buy])
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                 signal=signal.reset_index(drop=True))
