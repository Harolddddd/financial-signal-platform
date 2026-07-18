from __future__ import annotations
import pandas as pd
from catboost import CatBoostClassifier
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class CatBoostStrategy(Strategy):
    data_source = "features"
    handles_nan = True

    def __init__(self, iterations: int = 200, depth: int = 6, learning_rate: float = 0.1) -> None:
        self._model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        )
        self._classes: list[str] = []
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy()
        y = df["label"].to_numpy()
        self._model.fit(X, y)
        self._classes = [str(c) for c in self._model.classes_]

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].to_numpy()
        proba = self._model.predict_proba(X)
        if "Buy" not in self._classes:
            n = len(df)
            return PredictionResult(
                confidence=pd.Series([0.0] * n),
                signal=pd.Series(["Hold"] * n),
            )
        buy_idx = self._classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
