import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.zoo.logistic_regression import LogisticRegressionClassifier
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.zoo.xgboost_model import XGBoostClassifier
from src.models.zoo.lightgbm_model import LightGBMClassifier
from src.models.zoo.svm_model import SVMClassifier
from src.models.zoo.naive_bayes import NaiveBayesClassifier
from src.models.base_classifier import BaseClassifier

_CLASSES = ["Buy", "Hold", "Sell"]

ALL_MODELS = [
    LogisticRegressionClassifier,
    RandomForestClassifier_,
    XGBoostClassifier,
    LightGBMClassifier,
    SVMClassifier,
    NaiveBayesClassifier,
]


def _make_data(n: int = 300):
    X, y_int = make_classification(
        n_samples=n, n_features=22, n_classes=3,
        n_informative=10, n_redundant=5, random_state=42,
    )
    y = np.array([_CLASSES[i] for i in y_int])
    return X, y


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_is_base_classifier_subclass(ModelClass):
    assert issubclass(ModelClass, BaseClassifier)


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_fit_predict_cycle(ModelClass):
    X, y = _make_data()
    clf = ModelClass()
    clf.fit(X[:200], y[:200])
    preds = clf.predict(X[200:])
    assert preds.shape == (100,)
    assert set(preds).issubset(set(_CLASSES))


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_predict_proba_shape(ModelClass):
    X, y = _make_data()
    clf = ModelClass()
    clf.fit(X[:200], y[:200])
    proba = clf.predict_proba(X[200:])
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_name_is_string(ModelClass):
    assert isinstance(ModelClass().name, str)
    assert len(ModelClass().name) > 0


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_default_params_is_dict(ModelClass):
    assert isinstance(ModelClass().default_params, dict)
