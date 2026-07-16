from __future__ import annotations
import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class MLPStrategy(Strategy):
    data_source = "features"

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        max_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self._model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=False,
        )
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        y = df["label"].to_numpy()
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
