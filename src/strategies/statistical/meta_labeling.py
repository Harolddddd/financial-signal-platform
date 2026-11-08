from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from src.strategies.threads import N_JOBS
from src.strategies.base import Strategy, PredictionResult
from src.strategies.statistical.calibration_utils import (
    BinnedCalibrator, feature_columns, hold_result, is_buy, signals_from, time_split,
)


class MetaLabelingStrategy(Strategy):
    """Meta-labeling (López de Prado, *Advances in Financial ML*, ch. 3).

    A primary ExtraTrees model is run at a deliberately low Buy threshold
    (high recall). A secondary model, trained only on a later slice of the
    window the primary never saw, learns *which* primary Buys actually hit.
    Confidence = secondary P(hit), so it filters the primary's false
    positives instead of trusting its raw score.
    """
    data_source = "features"

    def __init__(
        self,
        primary_threshold: float = 0.4,
        meta_fraction: float = 0.3,
        buy_threshold: float = 0.5,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        self.primary_threshold = primary_threshold
        self.meta_fraction = meta_fraction
        self.buy_threshold = buy_threshold
        self._primary = ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", n_jobs=N_JOBS, random_state=random_state,
        )
        self._meta = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=200, l2_regularization=1.0, random_state=random_state,
        )
        # A raw secondary model is itself over-confident (backtest: "0.9" hit
        # ~42%), so its output is mapped to the hit rate seen on a final slice.
        self._calib = BinnedCalibrator()
        self._meta_fitted = False
        self._feature_cols: list[str] = []

    def _primary_buy_proba(self, X: np.ndarray) -> np.ndarray:
        classes = list(self._primary.classes_)
        if "Buy" not in classes:
            return np.zeros(len(X))
        return self._primary.predict_proba(X)[:, classes.index("Buy")]

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = feature_columns(df)
        early, late = time_split(df, self.meta_fraction)
        if len(early) == 0 or len(late) == 0:
            early, late = df, df.iloc[0:0]

        X_early = early[self._feature_cols].fillna(0.0).to_numpy()
        self._primary.fit(X_early, early["label"].to_numpy())

        self._meta_fitted = False
        meta_train, meta_cal = time_split(late, 0.5)
        meta_rows = [self._primary_buys(part) for part in (meta_train, meta_cal)]
        if any(len(y) < 50 or len(np.unique(y)) < 2 for _, y in meta_rows):
            return
        (X_mt, y_mt), (X_mc, y_mc) = meta_rows
        self._meta.fit(X_mt, y_mt)
        self._calib.fit(self._meta.predict_proba(X_mc)[:, 1], y_mc)
        self._meta_fitted = True

    def _primary_buys(self, part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Rows the primary flags as Buy: meta inputs (features + primary
        score) and whether each one actually hit."""
        if len(part) == 0:
            return np.empty((0, len(self._feature_cols) + 1)), np.empty(0)
        X = part[self._feature_cols].fillna(0.0).to_numpy()
        p = self._primary_buy_proba(X)
        mask = p >= self.primary_threshold
        return np.column_stack([X[mask], p[mask]]), is_buy(part)[mask]

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        if not self._meta_fitted:
            return hold_result(len(df))
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        p = self._primary_buy_proba(X)
        meta_p = self._calib.predict(self._meta.predict_proba(np.column_stack([X, p]))[:, 1])
        confidence = np.where(p >= self.primary_threshold, meta_p, 0.0)
        return signals_from(confidence, self.buy_threshold)
