import numpy as np
from sklearn.datasets import make_classification
from src.models.zoo.mlp_model import MLPClassifier_
from src.models.base_classifier import BaseClassifier

_CLASSES = ["Buy", "Hold", "Sell"]


def _make_data(n: int = 300):
    X, y_int = make_classification(
        n_samples=n, n_features=22, n_classes=3,
        n_informative=10, n_redundant=5, random_state=42,
    )
    return X, np.array([_CLASSES[i] for i in y_int])


def test_mlp_is_base_classifier():
    assert issubclass(MLPClassifier_, BaseClassifier)


def test_mlp_fit_predict():
    X, y = _make_data()
    clf = MLPClassifier_()
    clf.fit(X[:200], y[:200])
    preds = clf.predict(X[200:])
    assert preds.shape == (100,)
    assert set(preds).issubset(set(_CLASSES))


def test_mlp_predict_proba_sums_to_one():
    X, y = _make_data()
    clf = MLPClassifier_()
    clf.fit(X[:200], y[:200])
    proba = clf.predict_proba(X[200:])
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_mlp_name():
    assert MLPClassifier_().name == "mlp"
