import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from src.models.base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = SVC(**merged)
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "svm"

    @property
    def default_params(self) -> dict:
        return {"C": 1.0, "kernel": "rbf", "probability": True,
                "class_weight": "balanced", "random_state": 42}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scaler.transform(X))
