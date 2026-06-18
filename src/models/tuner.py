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
        "max_iter": 1000, "random_state": 42, "class_weight": "balanced",
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
        "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1,
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
