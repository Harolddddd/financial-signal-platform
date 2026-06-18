import numpy as np
import pytest
from src.models.base_classifier import BaseClassifier, EvaluationResult


class _ConcreteClassifier(BaseClassifier):
    @property
    def name(self) -> str:
        return "concrete"

    @property
    def default_params(self) -> dict:
        return {"param": 1}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(["Buy"] * len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile([0.7, 0.2, 0.1], (len(X), 1))


def test_concrete_subclass_satisfies_interface():
    clf = _ConcreteClassifier()
    X = np.random.randn(10, 5)
    y = np.array(["Buy", "Hold", "Sell"] * 3 + ["Buy"])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 10
    proba = clf.predict_proba(X)
    assert proba.shape == (10, 3)


def test_evaluation_result_is_dataclass():
    er = EvaluationResult(
        accuracy=0.6, precision_buy=0.7, recall_buy=0.5,
        f1_buy=0.58, f1_macro=0.55,
        confusion_matrix=[[5, 2, 1], [1, 4, 2], [0, 1, 4]],
        class_labels=["Buy", "Hold", "Sell"],
        n_samples=20,
    )
    assert er.precision_buy == 0.7
    assert er.n_samples == 20


def test_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseClassifier()
