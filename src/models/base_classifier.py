from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvaluationResult:
    accuracy: float
    precision_buy: float
    recall_buy: float
    f1_buy: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    n_samples: int


class BaseClassifier(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on X (n_samples, n_features) with string labels y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return string label predictions, shape (n_samples,)."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, n_classes)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier, e.g. 'random_forest'."""

    @property
    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Default hyperparameters for this model."""
