from __future__ import annotations
import logging

import numpy as np
import shap

from src.models.base_classifier import BaseClassifier
from src.signals.signal_engine import Signal

logger = logging.getLogger(__name__)

_TREE_MODELS   = {"random_forest", "xgboost", "lightgbm"}
_LINEAR_MODELS = {"logistic_regression"}
_BUY_IDX = 0  # Buy is class 0 alphabetically


def explain_prediction(
    model: BaseClassifier,
    X_row: np.ndarray,
    feature_cols: list[str],
    background: np.ndarray | None = None,
) -> dict[str, float]:
    if X_row.ndim == 1:
        X_row = X_row.reshape(1, -1)

    try:
        shap_values = _compute_shap(model, X_row, background)
    except Exception as e:
        logger.warning("SHAP failed for %s: %s — returning zeros", model.name, e)
        return {col: 0.0 for col in feature_cols}

    # Handle different SHAP return types:
    # - list of arrays (older shap / some tree models): list[array(n_samples, n_features)]
    # - 3D array (new shap Explanation unpacked): (n_samples, n_features, n_classes)
    # - 2D array: (n_samples, n_features) for binary / single output
    if isinstance(shap_values, list):
        vals = np.array(shap_values[_BUY_IDX])[0]
    elif hasattr(shap_values, "values"):
        # shap.Explanation object
        v = shap_values.values
        if v.ndim == 3:
            vals = v[0, :, _BUY_IDX]
        else:
            vals = v[0]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            vals = shap_values[0, :, _BUY_IDX]
        else:
            vals = shap_values[0]
    else:
        vals = np.zeros(len(feature_cols))

    explanation = dict(zip(feature_cols, [float(v) for v in vals]))
    return dict(sorted(explanation.items(), key=lambda kv: abs(kv[1]), reverse=True))


def _compute_shap(
    model: BaseClassifier,
    X_row: np.ndarray,
    background: np.ndarray | None,
) -> object:
    if model.name in _TREE_MODELS:
        inner = getattr(model, "_model", model)
        explainer = shap.TreeExplainer(inner)
        return explainer.shap_values(X_row)

    if model.name in _LINEAR_MODELS:
        inner = getattr(model, "_model", model)
        scaler = getattr(model, "_scaler", None)
        if background is not None:
            bg = scaler.transform(background[:50]) if scaler is not None else background[:50]
        else:
            bg = np.zeros((1, X_row.shape[1]))
        X_scaled = scaler.transform(X_row) if scaler is not None else X_row
        explainer = shap.LinearExplainer(inner, bg)
        return explainer.shap_values(X_scaled)

    bg = background[:50] if background is not None and len(background) >= 50 else background
    if bg is None:
        bg = np.zeros((1, X_row.shape[1]))

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(x)

    explainer = shap.KernelExplainer(predict_fn, bg)
    return explainer.shap_values(X_row, nsamples=100, silent=True)


def attach_explanations(
    signals: list[Signal],
    model: BaseClassifier,
    df: object,  # pl.DataFrame
    feature_cols: list[str],
) -> list[Signal]:
    latest = df.sort("time").group_by("ticker").tail(1)
    ticker_rows = {row["ticker"]: row for row in latest.iter_rows(named=True)}
    background = df.select(feature_cols).to_numpy()

    for signal in signals:
        row = ticker_rows.get(signal.ticker)
        if row is None:
            continue
        X_row = np.array([[row[c] for c in feature_cols]])
        signal.feature_explanation = explain_prediction(
            model, X_row, feature_cols, background=background
        )

    return signals
