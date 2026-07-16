from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}
_MAX_TRAIN_ROWS = 10_000  # SVM scales O(n²); cap to keep fold time manageable


class SVMStrategy(Strategy):
    data_source = "features"

    def __init__(self, kernel: str = "rbf", C: float = 1.0) -> None:
        self._model = SVC(
            kernel=kernel, C=C,
            probability=True, class_weight="balanced",
            max_iter=1000, random_state=42,
        )
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        y = df["label"].to_numpy()
        if len(X) > _MAX_TRAIN_ROWS:
            idx = np.random.default_rng(42).choice(len(X), _MAX_TRAIN_ROWS, replace=False)
            X, y = X[idx], y[idx]
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        proba = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
