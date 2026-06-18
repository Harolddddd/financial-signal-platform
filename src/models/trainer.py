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
