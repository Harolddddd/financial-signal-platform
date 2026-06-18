import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from src.models.base_classifier import BaseClassifier


class LightGBMClassifier(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = LGBMClassifier(**merged)
        self._le = LabelEncoder()

    @property
    def name(self) -> str:
        return "lightgbm"

    @property
    def default_params(self) -> dict:
        return {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
            "verbose": -1,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_enc = self._le.fit_transform(y)
        self._model.fit(X, y_enc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._le.inverse_transform(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
