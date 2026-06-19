# src/strategies/statistical/linear.py
from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class LinearStrategy(Strategy):
    data_source = "features"

    def __init__(self, buy_threshold: float = 0.005) -> None:
        self.buy_threshold = buy_threshold
        self._model = LinearRegression()
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = df["forward_return_5d"].to_numpy()
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].to_numpy()
        pred_return = self._model.predict(X)
        # Map predicted return [-10%, +10%] → confidence [0, 1]
        confidence = pd.Series(((pred_return + 0.10) / 0.20).clip(0, 1))
        signal = pd.Series(
            ["Buy" if r >= self.buy_threshold else "Hold" for r in pred_return]
        )
        return PredictionResult(confidence=confidence, signal=signal)
