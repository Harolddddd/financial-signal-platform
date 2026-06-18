import numpy as np
from sklearn.datasets import make_classification
import pytest

from src.models.tuner import tune
from src.models.zoo.logistic_regression import LogisticRegressionClassifier
from src.models.zoo.random_forest import RandomForestClassifier_

_CLASSES = ["Buy", "Hold", "Sell"]


def _make_data(n: int = 300):
    X, y_int = make_classification(
        n_samples=n, n_features=22, n_classes=3,
        n_informative=10, n_redundant=5, random_state=7,
    )
    return X, np.array([_CLASSES[i] for i in y_int])


def test_tune_returns_dict():
    X, y = _make_data(300)
    best = tune(
        "logistic_regression",
        LogisticRegressionClassifier,
        X[:200], y[:200],
        X[200:], y[200:],
        n_trials=3,
    )
    assert isinstance(best, dict)
    assert len(best) > 0


def test_tune_params_usable_to_instantiate_model():
    X, y = _make_data(300)
    best = tune(
        "random_forest",
        RandomForestClassifier_,
        X[:200], y[:200],
        X[200:], y[200:],
        n_trials=3,
    )
    clf = RandomForestClassifier_(**best)
    clf.fit(X[:200], y[:200])
    preds = clf.predict(X[200:])
    assert len(preds) == 100


def test_tune_raises_on_unknown_model():
    X, y = _make_data()
    with pytest.raises(ValueError, match="No param space"):
        tune("unknown_model", LogisticRegressionClassifier,
             X[:200], y[:200], X[200:], y[200:], n_trials=1)
