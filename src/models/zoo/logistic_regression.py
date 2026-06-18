import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = LogisticRegression(**merged)
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "logistic_regression"

    @property
    def default_params(self) -> dict:
        return {"C": 1.0, "max_iter": 1000, "random_state": 42,
                "class_weight": "balanced"}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scaler.transform(X))
