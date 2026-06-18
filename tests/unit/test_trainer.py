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
