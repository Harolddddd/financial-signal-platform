from __future__ import annotations
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class HistGradientBoostingStrategy(Strategy):
    data_source = "features"
    handles_nan = True

    def __init__(self, max_iter: int = 200, max_depth: int = 6, learning_rate: float = 0.1) -> None:
        self._model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        self._le = LabelEncoder()
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = self._le.fit_transform(df["label"].to_numpy())
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].to_numpy()
        proba = self._model.predict_proba(X)
        classes = list(self._le.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(
                confidence=pd.Series([0.0] * n),
                signal=pd.Series(["Hold"] * n),
            )
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
