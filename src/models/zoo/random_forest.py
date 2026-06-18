import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.models.base_classifier import BaseClassifier


class RandomForestClassifier_(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = RandomForestClassifier(**merged)

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def default_params(self) -> dict:
        return {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5,
                "class_weight": "balanced", "random_state": 42, "n_jobs": -1}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
