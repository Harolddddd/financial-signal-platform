import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest

from src.models.registry import save_model, load_model, list_models, ModelRecord
from src.models.zoo.random_forest import RandomForestClassifier_
from src.models.base_classifier import EvaluationResult
from sklearn.datasets import make_classification

_CLASSES = ["Buy", "Hold", "Sell"]
_FEATURE_COLS = [f"f{i}" for i in range(22)]
_EVAL = EvaluationResult(
    accuracy=0.55, precision_buy=0.62, recall_buy=0.48,
    f1_buy=0.54, f1_macro=0.50,
    confusion_matrix=[[5, 2, 1], [1, 4, 2], [0, 1, 4]],
    class_labels=["Buy", "Hold", "Sell"], n_samples=20,
)


def _trained_rf():
    X, y_int = make_classification(
        n_samples=200, n_features=22, n_classes=3,
        n_informative=10, n_redundant=5, random_state=42,
    )
    y = np.array([_CLASSES[i] for i in y_int])
    clf = RandomForestClassifier_(n_estimators=10)
    clf.fit(X, y)
    return clf


def test_save_creates_joblib_and_json(tmp_path):
    clf = _trained_rf()
    path = save_model(clf, _EVAL, clf.default_params, _FEATURE_COLS, tmp_path)
    assert path.exists()
    assert path.suffix == ".joblib"
    json_path = path.with_suffix(".json")
    assert json_path.exists()
    meta = json.loads(json_path.read_text())
    assert meta["model_name"] == "random_forest"


def test_load_model_returns_callable_classifier(tmp_path):
    clf = _trained_rf()
    save_model(clf, _EVAL, clf.default_params, _FEATURE_COLS, tmp_path)
    loaded = load_model("random_forest", tmp_path)
    X, _ = make_classification(n_samples=10, n_features=22, n_informative=10, n_redundant=5, random_state=1)
    preds = loaded.predict(X)
    assert len(preds) == 10


def test_load_model_latest_version_by_default(tmp_path):
    import time
    clf = _trained_rf()
    save_model(clf, _EVAL, clf.default_params, _FEATURE_COLS, tmp_path)
    time.sleep(0.01)
    save_model(clf, _EVAL, clf.default_params, _FEATURE_COLS, tmp_path)
    loaded = load_model("random_forest", tmp_path)
    assert loaded is not None


def test_list_models_returns_model_records(tmp_path):
    clf = _trained_rf()
    save_model(clf, _EVAL, clf.default_params, _FEATURE_COLS, tmp_path)
    records = list_models(tmp_path)
    assert len(records) >= 1
    assert isinstance(records[0], ModelRecord)
    assert records[0].model_name == "random_forest"


def test_load_model_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No saved versions"):
        load_model("nonexistent_model", tmp_path)
