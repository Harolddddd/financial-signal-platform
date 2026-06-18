import numpy as np
from sklearn.datasets import make_classification
from src.models.zoo.lstm_model import LSTMClassifier
from src.models.base_classifier import BaseClassifier

_CLASSES = ["Buy", "Hold", "Sell"]


def _make_data(n: int = 300):
    X, y_int = make_classification(
        n_samples=n, n_features=22, n_classes=3,
        n_informative=10, n_redundant=5, random_state=42,
    )
    return X, np.array([_CLASSES[i] for i in y_int])


def test_lstm_is_base_classifier():
    assert issubclass(LSTMClassifier, BaseClassifier)


def test_lstm_fit_predict():
    X, y = _make_data(200)
    clf = LSTMClassifier(seq_len=10, hidden_size=32, n_layers=1, epochs=2)
    clf.fit(X[:150], y[:150])
    preds = clf.predict(X[150:])
    assert preds.shape == (50,)
    assert set(preds).issubset(set(_CLASSES))


def test_lstm_predict_proba_shape():
    X, y = _make_data(200)
    clf = LSTMClassifier(seq_len=10, hidden_size=32, n_layers=1, epochs=2)
    clf.fit(X[:150], y[:150])
    proba = clf.predict_proba(X[150:])
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_lstm_predict_before_fit_raises():
    import pytest
    clf = LSTMClassifier(seq_len=10)
    with pytest.raises(RuntimeError, match="not trained"):
        clf.predict(np.random.randn(5, 22))


def test_lstm_name():
    assert LSTMClassifier().name == "lstm"
