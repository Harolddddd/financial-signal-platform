import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from src.models.base_classifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = GaussianNB(**{k: v for k, v in merged.items() if k != "random_state"})
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "naive_bayes"

    @property
    def default_params(self) -> dict:
        return {"var_smoothing": 1e-9}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scaler.transform(X))
