# Financial Platform — Plan 3: ML Classification Hub

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pluggable, precision-focused multi-model classification hub — a shared interface, a model zoo of 8 classifiers, a walk-forward training pipeline, Optuna hyperparameter tuning, and a model registry — that reads from the feature store built in Plan 2 and outputs trained, versioned models ready for Plan 4's backtesting engine.

**Architecture:** A `BaseClassifier` abstract class defines the interface all models satisfy. Each zoo model is a thin wrapper that handles its own preprocessing (scaling for LR/SVM/MLP, sequence creation for LSTM). The `Trainer` performs walk-forward cross-validation: it splits the feature DataFrame by date into rolling train/test folds, fits the model on each fold, and collects `EvaluationResult` per fold. The `Tuner` wraps a single train/val split in an Optuna study, optimizing for precision on the "Buy" class. The `Registry` persists trained models to disk with joblib alongside a JSON metadata file.

**Tech Stack:** Python 3.11, scikit-learn 1.5, XGBoost 2.0, LightGBM 4.3, PyTorch 2.3, Optuna 3.6, joblib 1.4, Polars 0.20, pytest 8.x

**Dependency on Plans 1 & 2:** Reads the `features` table and Parquet exports from Plan 2. Imports `src.features.duckdb_client.load_training_data` and `src.ingestion.db`.

## Global Constraints

- Python >= 3.11; all function signatures require type hints
- `BaseClassifier.fit` receives raw unscaled numpy arrays with string labels (`"Buy"` / `"Hold"` / `"Sell"`); each model handles its own internal preprocessing
- Models that require scaling (LR, SVM, MLP) include an internal `StandardScaler` that is fit on training data only
- LSTM handles sequence creation internally — the trainer always provides flat 2D feature arrays
- Walk-forward trainer never allows test-fold data to influence train-fold fitting (no look-ahead)
- Precision for the `"Buy"` class is the primary optimization metric; accuracy is secondary
- Registry saves one `.joblib` + one `.json` per model per version; version = ISO-8601 timestamp
- Tests use pytest only; synthetic data via `sklearn.datasets.make_classification` — no real market data required

## Feature Columns (from Plan 2)

```python
FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]
LABEL_COL = "label"
CLASSES = ["Buy", "Hold", "Sell"]
```

---

### Task 1: Base Classifier Interface & Dependency Update

**Files:**
- Modify: `pyproject.toml`
- Create: `src/models/__init__.py`
- Create: `src/models/zoo/__init__.py`
- Create: `src/models/base_classifier.py`
- Test: `tests/unit/test_base_classifier.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `BaseClassifier` — abstract class with `fit`, `predict`, `predict_proba`, `name`, `default_params`
  - `EvaluationResult` dataclass — `accuracy`, `precision_buy`, `recall_buy`, `f1_buy`, `f1_macro`, `confusion_matrix`, `class_labels`, `n_samples`

- [ ] **Step 1: Add ML dependencies to pyproject.toml**

Open `pyproject.toml` and add to `dependencies`:

```toml
    "scikit-learn>=1.5.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.3.0",
    "optuna>=3.6.0",
    "joblib>=1.4.0",
```

(torch is already present from Plan 1's sentiment processor.)

- [ ] **Step 2: Install updated dependencies**

```bash
pip install scikit-learn>=1.5.0 xgboost>=2.0.0 lightgbm>=4.3.0 optuna>=3.6.0 joblib>=1.4.0
```
Expected: All packages install without errors.

- [ ] **Step 3: Write failing test**

```python
# tests/unit/test_base_classifier.py
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
```

- [ ] **Step 4: Run test to verify it fails**

```bash
pytest tests/unit/test_base_classifier.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 5: Create src/models/base_classifier.py**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
```

- [ ] **Step 6: Create empty __init__ files**

```bash
touch src/models/__init__.py src/models/zoo/__init__.py
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/unit/test_base_classifier.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/models/ tests/unit/test_base_classifier.py
git commit -m "feat: BaseClassifier interface and EvaluationResult dataclass"
```

---

### Task 2: Classical Classifiers Zoo

**Files:**
- Create: `src/models/zoo/logistic_regression.py`
- Create: `src/models/zoo/random_forest.py`
- Create: `src/models/zoo/xgboost_model.py`
- Create: `src/models/zoo/lightgbm_model.py`
- Create: `src/models/zoo/svm_model.py`
- Create: `src/models/zoo/naive_bayes.py`
- Test: `tests/unit/test_zoo_classical.py`

**Interfaces:**
- Consumes: `BaseClassifier`
- Produces: Six classes all satisfying `BaseClassifier`:
  - `LogisticRegressionClassifier(name="logistic_regression")`
  - `RandomForestClassifier_(name="random_forest")`
  - `XGBoostClassifier(name="xgboost")`
  - `LightGBMClassifier(name="lightgbm")`
  - `SVMClassifier(name="svm")`
  - `NaiveBayesClassifier(name="naive_bayes")`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_zoo_classical.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_zoo_classical.py -v
```
Expected: `ModuleNotFoundError` for each zoo module.

- [ ] **Step 3: Implement src/models/zoo/logistic_regression.py**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = LogisticRegression(**merged)
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "logistic_regression"

    @property
    def default_params(self) -> dict:
        return {"C": 1.0, "max_iter": 1000, "random_state": 42,
                "class_weight": "balanced", "multi_class": "multinomial"}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scaler.transform(X))
```

- [ ] **Step 4: Implement src/models/zoo/random_forest.py**

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.models.base_classifier import BaseClassifier


class RandomForestClassifier_(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = RandomForestClassifier(**merged)

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def default_params(self) -> dict:
        return {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5,
                "class_weight": "balanced", "random_state": 42, "n_jobs": -1}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
```

- [ ] **Step 5: Implement src/models/zoo/xgboost_model.py**

```python
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from src.models.base_classifier import BaseClassifier


class XGBoostClassifier(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = XGBClassifier(**merged)
        self._le = LabelEncoder()

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def default_params(self) -> dict:
        return {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "use_label_encoder": False, "eval_metric": "mlogloss",
            "random_state": 42, "n_jobs": -1,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y_enc = self._le.fit_transform(y)
        self._model.fit(X, y_enc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._le.inverse_transform(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
```

- [ ] **Step 6: Implement src/models/zoo/lightgbm_model.py**

```python
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
```

- [ ] **Step 7: Implement src/models/zoo/svm_model.py**

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from src.models.base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):

    def __init__(self, **params):
        merged = {**self.default_params, **params}
        self._model = SVC(**merged)
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "svm"

    @property
    def default_params(self) -> dict:
        return {"C": 1.0, "kernel": "rbf", "probability": True,
                "class_weight": "balanced", "random_state": 42}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scaler.transform(X))
```

- [ ] **Step 8: Implement src/models/zoo/naive_bayes.py**

```python
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
```

- [ ] **Step 9: Run tests to verify they pass**

```bash
pytest tests/unit/test_zoo_classical.py -v
```
Expected: 30 tests PASS (5 tests × 6 models via parametrize).

- [ ] **Step 10: Commit**

```bash
git add src/models/zoo/ tests/unit/test_zoo_classical.py
git commit -m "feat: classical classifier zoo (LR, RF, XGBoost, LightGBM, SVM, NaiveBayes)"
```

---

### Task 3: MLP Classifier

**Files:**
- Create: `src/models/zoo/mlp_model.py`
- Test: `tests/unit/test_zoo_mlp.py`

**Interfaces:**
- Consumes: `BaseClassifier`
- Produces:
  - `MLPClassifier_(name="mlp")` — wraps `sklearn.neural_network.MLPClassifier` with internal `StandardScaler`
  - `default_params`: `hidden_layer_sizes=(128, 64)`, `activation="relu"`, `max_iter=300`, `early_stopping=True`, `random_state=42`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_zoo_mlp.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_zoo_mlp.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/models/zoo/mlp_model.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_zoo_mlp.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/zoo/mlp_model.py tests/unit/test_zoo_mlp.py
git commit -m "feat: MLP classifier zoo wrapper"
```

---

### Task 4: LSTM Classifier

**Files:**
- Create: `src/models/zoo/lstm_model.py`
- Test: `tests/unit/test_zoo_lstm.py`

**Interfaces:**
- Consumes: `BaseClassifier`; PyTorch
- Produces:
  - `LSTMClassifier(name="lstm")`
  - `default_params`: `seq_len=20`, `hidden_size=64`, `n_layers=2`, `dropout=0.2`, `lr=1e-3`, `epochs=30`, `batch_size=64`
  - Internally calls `_make_sequences(X, y_enc, seq_len)` to convert flat arrays into `(n, seq_len, n_features)` tensors
  - `predict_proba` pads short inputs (len < seq_len) with zeros; returns `(n_samples, 3)` — same shape as all other classifiers

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_zoo_lstm.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_zoo_lstm.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/models/zoo/lstm_model.py**

```python
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.models.base_classifier import BaseClassifier


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int,
                 n_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMClassifier(BaseClassifier):

    def __init__(self, seq_len: int = 20, hidden_size: int = 64, n_layers: int = 2,
                 dropout: float = 0.2, lr: float = 1e-3, epochs: int = 30,
                 batch_size: int = 64):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._net: _LSTMNet | None = None
        self._le = LabelEncoder()
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "lstm"

    @property
    def default_params(self) -> dict:
        return {
            "seq_len": 20, "hidden_size": 64, "n_layers": 2,
            "dropout": 0.2, "lr": 1e-3, "epochs": 30, "batch_size": 64,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_s = self._scaler.fit_transform(X)
        y_enc = self._le.fit_transform(y)
        n_classes = len(self._le.classes_)

        X_seq, y_seq = _make_sequences(X_s, y_enc, self.seq_len)
        if len(X_seq) == 0:
            raise ValueError(f"Need > {self.seq_len} samples, got {len(X)}")

        self._net = _LSTMNet(X_seq.shape[2], self.hidden_size, self.n_layers,
                             n_classes, self.dropout)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        self._net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(self._net(xb), yb).backward()
                optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not trained — call fit() first")
        X_s = self._scaler.transform(X)
        X_seq, _ = _make_sequences(X_s, np.zeros(len(X_s), dtype=int), self.seq_len)
        if len(X_seq) == 0:
            n_classes = len(self._le.classes_)
            return np.full((len(X), n_classes), 1.0 / n_classes)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.tensor(X_seq, dtype=torch.float32))
            proba_seq = torch.softmax(logits, dim=1).numpy()
        n_classes = proba_seq.shape[1]
        out = np.full((len(X), n_classes), 1.0 / n_classes)
        out[self.seq_len:] = proba_seq
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not trained — call fit() first")
        proba = self.predict_proba(X)
        return self._le.inverse_transform(proba.argmax(axis=1))


def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X) - seq_len
    if n <= 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0, dtype=int)
    X_seq = np.stack([X[i: i + seq_len] for i in range(n)])
    y_seq = y[seq_len:]
    return X_seq, y_seq
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_zoo_lstm.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/zoo/lstm_model.py tests/unit/test_zoo_lstm.py
git commit -m "feat: LSTM classifier with internal sequence creation and PyTorch training loop"
```

---

### Task 5: Evaluator

**Files:**
- Create: `src/models/evaluator.py`
- Test: `tests/unit/test_evaluator.py`

**Interfaces:**
- Consumes: `y_true: np.ndarray`, `y_pred: np.ndarray` — both string arrays
- Produces:
  - `evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult`
    — computes accuracy, precision/recall/F1 for `"Buy"` class, macro F1, confusion matrix
  - `EvaluationResult` (re-exported from `base_classifier`)

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_evaluator.py
import numpy as np
import pytest
from src.models.evaluator import evaluate
from src.models.base_classifier import EvaluationResult


def test_evaluate_perfect_predictions():
    y = np.array(["Buy", "Hold", "Sell", "Buy", "Buy"])
    result = evaluate(y, y)
    assert result.accuracy == 1.0
    assert result.precision_buy == 1.0
    assert result.recall_buy == 1.0
    assert result.f1_buy == 1.0


def test_evaluate_returns_evaluation_result():
    y_true = np.array(["Buy", "Hold", "Sell", "Buy", "Hold"])
    y_pred = np.array(["Buy", "Sell", "Sell", "Hold", "Hold"])
    result = evaluate(y_true, y_pred)
    assert isinstance(result, EvaluationResult)
    assert result.n_samples == 5


def test_confusion_matrix_shape():
    y_true = np.array(["Buy", "Hold", "Sell"] * 10)
    y_pred = np.array(["Buy", "Hold", "Buy"] * 10)
    result = evaluate(y_true, y_pred)
    assert len(result.confusion_matrix) == 3
    assert all(len(row) == 3 for row in result.confusion_matrix)


def test_class_labels_sorted():
    y_true = np.array(["Sell", "Buy", "Hold"])
    y_pred = np.array(["Sell", "Buy", "Hold"])
    result = evaluate(y_true, y_pred)
    assert result.class_labels == ["Buy", "Hold", "Sell"]


def test_all_hold_predictions_buy_precision_zero():
    y_true = np.array(["Buy", "Buy", "Hold", "Hold"])
    y_pred = np.array(["Hold", "Hold", "Hold", "Hold"])
    result = evaluate(y_true, y_pred)
    assert result.precision_buy == 0.0
    assert result.recall_buy == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_evaluator.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/models/evaluator.py**

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.models.base_classifier import EvaluationResult

_CLASSES = ["Buy", "Hold", "Sell"]


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> EvaluationResult:
    labels = sorted(set(y_true) | set(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    prec_buy = float(precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )[labels.index("Buy")] if "Buy" in labels else 0.0)
    rec_buy = float(recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )[labels.index("Buy")] if "Buy" in labels else 0.0)
    f1_buy = float(f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )[labels.index("Buy")] if "Buy" in labels else 0.0)
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return EvaluationResult(
        accuracy=accuracy,
        precision_buy=prec_buy,
        recall_buy=rec_buy,
        f1_buy=f1_buy,
        f1_macro=f1_macro,
        confusion_matrix=cm,
        class_labels=labels,
        n_samples=len(y_true),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_evaluator.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/evaluator.py tests/unit/test_evaluator.py
git commit -m "feat: precision-focused evaluator with Buy-class metrics"
```

---

### Task 6: Walk-Forward Trainer

**Files:**
- Create: `src/models/trainer.py`
- Test: `tests/unit/test_trainer.py`

**Interfaces:**
- Consumes: `BaseClassifier`, `evaluate`, feature `pl.DataFrame` from Plan 2
- Produces:
  - `FoldResult` dataclass: `fold: int`, `train_start: str`, `train_end: str`, `test_start: str`, `test_end: str`, `evaluation: EvaluationResult`
  - `WalkForwardResult` dataclass: `folds: list[FoldResult]`, `mean_precision_buy: float`, `mean_f1_macro: float`, `mean_accuracy: float`
  - `walk_forward_train(df: pl.DataFrame, model: BaseClassifier, feature_cols: list[str], label_col: str = "label", train_window_days: int = 500, test_window_days: int = 21, step_days: int = 21, min_train_samples: int = 100) -> WalkForwardResult`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_trainer.py
from datetime import datetime, timezone, timedelta
import numpy as np
import polars as pl
import pytest

from src.models.trainer import walk_forward_train, WalkForwardResult, FoldResult
from src.models.zoo.random_forest import RandomForestClassifier_

_FEATURE_COLS = ["f1", "f2", "f3"]


def _make_feature_df(n: int = 600) -> pl.DataFrame:
    import random
    random.seed(42)
    rng = np.random.default_rng(42)
    base = datetime(2021, 1, 4, tzinfo=timezone.utc)
    labels = ["Buy", "Hold", "Sell"]
    return pl.DataFrame({
        "time":   [base + timedelta(days=i) for i in range(n)],
        "ticker": ["AAPL"] * n,
        "f1": rng.standard_normal(n).tolist(),
        "f2": rng.standard_normal(n).tolist(),
        "f3": rng.standard_normal(n).tolist(),
        "label": [labels[i % 3] for i in range(n)],
    })


def test_walk_forward_returns_result():
    df = _make_feature_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_train(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    assert isinstance(result, WalkForwardResult)
    assert len(result.folds) >= 1


def test_each_fold_is_fold_result():
    df = _make_feature_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_train(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    for fold in result.folds:
        assert isinstance(fold, FoldResult)
        assert fold.evaluation.n_samples > 0


def test_mean_precision_is_average_of_folds():
    df = _make_feature_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_train(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    manual_mean = sum(f.evaluation.precision_buy for f in result.folds) / len(result.folds)
    assert abs(result.mean_precision_buy - manual_mean) < 1e-9


def test_test_window_never_overlaps_train_window():
    df = _make_feature_df()
    model = RandomForestClassifier_(n_estimators=10)
    result = walk_forward_train(
        df, model, _FEATURE_COLS,
        train_window_days=300, test_window_days=30, step_days=30,
    )
    for fold in result.folds:
        assert fold.test_start > fold.train_end
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_trainer.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/models/trainer.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

import numpy as np
import polars as pl

from src.models.base_classifier import BaseClassifier, EvaluationResult
from src.models.evaluator import evaluate

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    evaluation: EvaluationResult


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    mean_precision_buy: float
    mean_f1_macro: float
    mean_accuracy: float


def walk_forward_train(
    df: pl.DataFrame,
    model: BaseClassifier,
    feature_cols: list[str],
    label_col: str = "label",
    train_window_days: int = 500,
    test_window_days: int = 21,
    step_days: int = 21,
    min_train_samples: int = 100,
) -> WalkForwardResult:
    df_clean = df.drop_nulls(subset=feature_cols + [label_col]).sort("time")
    times = df_clean["time"].to_list()
    if not times:
        raise ValueError("DataFrame is empty after dropping nulls")

    t_start = times[0]
    t_end = times[-1]
    folds: list[FoldResult] = []
    fold_idx = 0

    cursor = t_start + timedelta(days=train_window_days)
    while cursor + timedelta(days=test_window_days) <= t_end + timedelta(days=1):
        train_start = cursor - timedelta(days=train_window_days)
        train_end = cursor - timedelta(days=1)
        test_start = cursor
        test_end = cursor + timedelta(days=test_window_days - 1)

        train_df = df_clean.filter(
            (pl.col("time") >= train_start) & (pl.col("time") <= train_end)
        )
        test_df = df_clean.filter(
            (pl.col("time") >= test_start) & (pl.col("time") <= test_end)
        )

        if len(train_df) < min_train_samples or len(test_df) == 0:
            cursor += timedelta(days=step_days)
            continue

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df[label_col].to_numpy()
        X_test = test_df.select(feature_cols).to_numpy()
        y_test = test_df[label_col].to_numpy()

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            eval_result = evaluate(y_test, y_pred)
        except Exception as e:
            logger.warning("Fold %d failed: %s", fold_idx, e)
            cursor += timedelta(days=step_days)
            continue

        folds.append(FoldResult(
            fold=fold_idx,
            train_start=train_start.isoformat(),
            train_end=train_end.isoformat(),
            test_start=test_start.isoformat(),
            test_end=test_end.isoformat(),
            evaluation=eval_result,
        ))
        fold_idx += 1
        cursor += timedelta(days=step_days)

    if not folds:
        raise ValueError("No valid folds produced — check data range and window sizes")

    n = len(folds)
    return WalkForwardResult(
        folds=folds,
        mean_precision_buy=sum(f.evaluation.precision_buy for f in folds) / n,
        mean_f1_macro=sum(f.evaluation.f1_macro for f in folds) / n,
        mean_accuracy=sum(f.evaluation.accuracy for f in folds) / n,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_trainer.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/trainer.py tests/unit/test_trainer.py
git commit -m "feat: walk-forward trainer with rolling date-based folds"
```

---

### Task 7: Optuna Tuner

**Files:**
- Create: `src/models/tuner.py`
- Test: `tests/unit/test_tuner.py`

**Interfaces:**
- Consumes: `BaseClassifier`, `evaluate`; Optuna
- Produces:
  - `tune(model_name: str, model_class: type[BaseClassifier], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 50) -> dict`
    — runs an Optuna study maximizing `precision_buy` on the validation split; returns best params dict
  - Param spaces defined for all 8 model names: `"logistic_regression"`, `"random_forest"`, `"xgboost"`, `"lightgbm"`, `"svm"`, `"naive_bayes"`, `"mlp"`, `"lstm"`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tuner.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_tuner.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/models/tuner.py**

```python
from __future__ import annotations
import logging
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.models.base_classifier import BaseClassifier
from src.models.evaluator import evaluate

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _lr_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "max_iter": 1000, "random_state": 42,
        "class_weight": "balanced", "multi_class": "multinomial",
    }


def _rf_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
    }


def _xgb_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "use_label_encoder": False, "eval_metric": "mlogloss",
        "random_state": 42, "n_jobs": -1,
    }


def _lgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1,
    }


def _svm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        "probability": True, "class_weight": "balanced", "random_state": 42,
    }


def _nb_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-5, log=True),
    }


def _mlp_space(trial: optuna.Trial) -> dict[str, Any]:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_categorical("layer_size", [32, 64, 128, 256])
    return {
        "hidden_layer_sizes": tuple([layer_size] * n_layers),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "max_iter": 300, "early_stopping": True,
        "validation_fraction": 0.1, "random_state": 42,
    }


def _lstm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "seq_len": trial.suggest_int("seq_len", 5, 30),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": 20,
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


_PARAM_SPACES: dict[str, Any] = {
    "logistic_regression": _lr_space,
    "random_forest": _rf_space,
    "xgboost": _xgb_space,
    "lightgbm": _lgbm_space,
    "svm": _svm_space,
    "naive_bayes": _nb_space,
    "mlp": _mlp_space,
    "lstm": _lstm_space,
}


def tune(
    model_name: str,
    model_class: type[BaseClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict[str, Any]:
    if model_name not in _PARAM_SPACES:
        raise ValueError(f"No param space for '{model_name}'. "
                         f"Available: {list(_PARAM_SPACES)}")

    space_fn = _PARAM_SPACES[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        clf = model_class(**params)
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            return evaluate(y_val, y_pred).precision_buy
        except Exception as e:
            logger.debug("Trial failed: %s", e)
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_tuner.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/tuner.py tests/unit/test_tuner.py
git commit -m "feat: Optuna tuner with per-model hyperparameter spaces, maximize Buy precision"
```

---

### Task 8: Model Registry

**Files:**
- Create: `src/models/registry.py`
- Test: `tests/unit/test_registry.py`

**Interfaces:**
- Consumes: `BaseClassifier`, `EvaluationResult`; joblib
- Produces:
  - `ModelRecord` dataclass: `model_name: str`, `version: str`, `params: dict`, `evaluation: EvaluationResult`, `feature_cols: list[str]`, `trained_at: str`
  - `save_model(model: BaseClassifier, evaluation: EvaluationResult, params: dict, feature_cols: list[str], registry_dir: Path) -> Path`
    — saves `{registry_dir}/{model.name}/{version}.joblib` and `{version}.json`; returns the `.joblib` path
  - `load_model(model_name: str, registry_dir: Path, version: str | None = None) -> BaseClassifier`
    — loads latest version if `version=None`
  - `list_models(registry_dir: Path) -> list[ModelRecord]`
    — returns all saved models sorted by `trained_at` descending

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_registry.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_registry.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/models/registry.py**

```python
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import joblib

from src.models.base_classifier import BaseClassifier, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ModelRecord:
    model_name: str
    version: str
    params: dict[str, Any]
    evaluation: EvaluationResult
    feature_cols: list[str]
    trained_at: str


def save_model(
    model: BaseClassifier,
    evaluation: EvaluationResult,
    params: dict[str, Any],
    feature_cols: list[str],
    registry_dir: Path,
) -> Path:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    model_dir = registry_dir / model.name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = model_dir / f"{version}.joblib"
    json_path = model_dir / f"{version}.json"

    joblib.dump(model, joblib_path)
    meta = {
        "model_name": model.name,
        "version": version,
        "params": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                   for k, v in params.items()},
        "evaluation": {
            "accuracy": evaluation.accuracy,
            "precision_buy": evaluation.precision_buy,
            "recall_buy": evaluation.recall_buy,
            "f1_buy": evaluation.f1_buy,
            "f1_macro": evaluation.f1_macro,
            "confusion_matrix": evaluation.confusion_matrix,
            "class_labels": evaluation.class_labels,
            "n_samples": evaluation.n_samples,
        },
        "feature_cols": feature_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s v%s → %s", model.name, version, joblib_path)
    return joblib_path


def load_model(
    model_name: str,
    registry_dir: Path,
    version: str | None = None,
) -> BaseClassifier:
    model_dir = registry_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No saved versions for '{model_name}' in {registry_dir}")

    versions = sorted(model_dir.glob("*.joblib"))
    if not versions:
        raise FileNotFoundError(f"No saved versions for '{model_name}' in {registry_dir}")

    if version:
        path = model_dir / f"{version}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Version {version} not found for '{model_name}'")
    else:
        path = versions[-1]

    return joblib.load(path)


def list_models(registry_dir: Path) -> list[ModelRecord]:
    records: list[ModelRecord] = []
    for json_path in sorted(registry_dir.rglob("*.json"), reverse=True):
        try:
            meta = json.loads(json_path.read_text())
            ev = meta["evaluation"]
            records.append(ModelRecord(
                model_name=meta["model_name"],
                version=meta["version"],
                params=meta["params"],
                evaluation=EvaluationResult(
                    accuracy=ev["accuracy"],
                    precision_buy=ev["precision_buy"],
                    recall_buy=ev["recall_buy"],
                    f1_buy=ev["f1_buy"],
                    f1_macro=ev["f1_macro"],
                    confusion_matrix=ev["confusion_matrix"],
                    class_labels=ev["class_labels"],
                    n_samples=ev["n_samples"],
                ),
                feature_cols=meta["feature_cols"],
                trained_at=meta["trained_at"],
            ))
        except Exception as e:
            logger.warning("Skipping malformed record %s: %s", json_path, e)
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_registry.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/registry.py tests/unit/test_registry.py
git commit -m "feat: model registry with joblib persistence, versioning, and metadata"
```

---

### Task 9: Model Retrain Airflow DAG

**Files:**
- Create: `dags/model_retrain_dag.py`
- Test: `tests/unit/test_model_retrain_dag.py`

**Interfaces:**
- Consumes: `load_training_data`, `walk_forward_train`, `tune`, `save_model`, all zoo classifiers
- Produces: DAG `model_retrain_dag`, `schedule_interval="@weekly"`, tasks: `tune_and_train_models`, `evaluate_and_register`; runs after `feature_engineering_dag`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_model_retrain_dag.py
import importlib


def test_model_retrain_dag_loads():
    mod = importlib.import_module("dags.model_retrain_dag")
    assert hasattr(mod, "dag")
    assert mod.dag.dag_id == "model_retrain_dag"


def test_dag_has_expected_tasks():
    mod = importlib.import_module("dags.model_retrain_dag")
    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "tune_and_train_models" in task_ids
    assert "evaluate_and_register" in task_ids


def test_evaluate_depends_on_tune():
    mod = importlib.import_module("dags.model_retrain_dag")
    dag = mod.dag
    tune_task = dag.get_task("tune_and_train_models")
    eval_task = dag.get_task("evaluate_and_register")
    assert eval_task.task_id in {d.task_id for d in tune_task.downstream_list}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_model_retrain_dag.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement dags/model_retrain_dag.py**

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

_default_args = {
    "owner": "platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

_PARQUET_DIR  = Path("/opt/airflow/data/features")
_REGISTRY_DIR = Path("/opt/airflow/data/registry")
_FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]

_MODEL_ZOO = [
    ("logistic_regression", "src.models.zoo.logistic_regression.LogisticRegressionClassifier"),
    ("random_forest",        "src.models.zoo.random_forest.RandomForestClassifier_"),
    ("xgboost",              "src.models.zoo.xgboost_model.XGBoostClassifier"),
    ("lightgbm",             "src.models.zoo.lightgbm_model.LightGBMClassifier"),
    ("svm",                  "src.models.zoo.svm_model.SVMClassifier"),
    ("naive_bayes",          "src.models.zoo.naive_bayes.NaiveBayesClassifier"),
    ("mlp",                  "src.models.zoo.mlp_model.MLPClassifier_"),
    ("lstm",                 "src.models.zoo.lstm_model.LSTMClassifier"),
]


def _tune_and_train(**context):
    import importlib
    import logging
    import numpy as np
    from src.features.duckdb_client import load_training_data
    from src.models.tuner import tune

    log = logging.getLogger(__name__)
    df = load_training_data(_PARQUET_DIR)
    df_clean = df.drop_nulls(subset=_FEATURE_COLS + ["label"])

    n = len(df_clean)
    split = int(n * 0.8)
    train_df = df_clean[:split]
    val_df = df_clean[split:]

    X_train = train_df.select(_FEATURE_COLS).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_val = val_df.select(_FEATURE_COLS).to_numpy()
    y_val = val_df["label"].to_numpy()

    best_params: dict[str, dict] = {}
    for model_name, model_path in _MODEL_ZOO:
        try:
            module_path, class_name = model_path.rsplit(".", 1)
            ModelClass = getattr(importlib.import_module(module_path), class_name)
            params = tune(model_name, ModelClass, X_train, y_train, X_val, y_val, n_trials=50)
            best_params[model_name] = {"path": model_path, "params": params}
            log.info("Tuned %s: %s", model_name, params)
        except Exception as e:
            log.error("Tune failed %s: %s", model_name, e)

    context["ti"].xcom_push(key="best_params", value=best_params)


def _evaluate_and_register(**context):
    import importlib
    import logging
    from src.features.duckdb_client import load_training_data
    from src.models.trainer import walk_forward_train
    from src.models.registry import save_model

    log = logging.getLogger(__name__)
    best_params = context["ti"].xcom_pull(key="best_params", task_ids="tune_and_train_models") or {}
    df = load_training_data(_PARQUET_DIR)

    for model_name, info in best_params.items():
        try:
            module_path, class_name = info["path"].rsplit(".", 1)
            ModelClass = getattr(importlib.import_module(module_path), class_name)
            clf = ModelClass(**info["params"])
            result = walk_forward_train(
                df, clf, _FEATURE_COLS,
                train_window_days=500, test_window_days=21, step_days=21,
            )
            last_eval = result.folds[-1].evaluation
            save_model(clf, last_eval, info["params"], _FEATURE_COLS, _REGISTRY_DIR)
            log.info("Registered %s — precision_buy=%.3f", model_name, last_eval.precision_buy)
        except Exception as e:
            log.error("Register failed %s: %s", model_name, e)


with DAG(
    dag_id="model_retrain_dag",
    default_args=_default_args,
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["models", "training"],
) as dag:
    tune_task = PythonOperator(
        task_id="tune_and_train_models",
        python_callable=_tune_and_train,
    )
    eval_task = PythonOperator(
        task_id="evaluate_and_register",
        python_callable=_evaluate_and_register,
    )
    tune_task >> eval_task
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_model_retrain_dag.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add dags/model_retrain_dag.py tests/unit/test_model_retrain_dag.py
git commit -m "feat: weekly model retrain DAG with Optuna tuning and registry registration"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Logistic Regression baseline | Task 2 |
| Random Forest (ensemble) | Task 2 |
| XGBoost / LightGBM (gradient boosting) | Task 2 |
| SVM (margin-based) | Task 2 |
| Naive Bayes (sentiment-dominant) | Task 2 |
| Neural Network MLP | Task 3 |
| LSTM / GRU (sequential) | Task 4 |
| Precision-focused evaluation (precision, F1, confusion matrix) | Task 5 |
| Walk-forward validation with rolling windows | Task 6 |
| Hyperparameter tuning via Optuna (Bayesian optimization) | Task 7 |
| Feature selection (configurable `feature_cols` param in trainer) | Tasks 6 + 9 |
| Model persistence and versioning | Task 8 |
| Pluggable architecture (add new classifier by implementing `BaseClassifier`) | Task 1 |
| Automated weekly retraining | Task 9 |

**Placeholder scan:** No TBDs. All steps include full code. LSTM test uses `epochs=2` for speed; production uses `epochs=30` via `default_params`.

**Type consistency:** `EvaluationResult` defined in `base_classifier.py` and re-used verbatim in `evaluator.py`, `trainer.py`, and `registry.py`. `FoldResult.evaluation` and `ModelRecord.evaluation` are both typed as `EvaluationResult`. `BaseClassifier.fit(X: np.ndarray, y: np.ndarray)` — all zoo models accept the same signature.

**GRU note:** The spec lists "LSTM / GRU". The `_LSTMNet` in Task 4 uses `nn.LSTM`. Switching to GRU requires replacing `nn.LSTM` with `nn.GRU` — the rest of the code is identical. Add `GRUClassifier` as a copy of `LSTMClassifier` with `nn.GRU` and `name = "gru"` when needed.
