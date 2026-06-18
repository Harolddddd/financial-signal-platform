import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.models.base_classifier import BaseClassifier


class MLPClassifier_(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = MLPClassifier(**merged)
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "mlp"

    @property
    def default_params(self) -> dict:
        return {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scaler.transform(X))
