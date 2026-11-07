from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.strategies.base import Strategy, PredictionResult
from src.strategies.statistical.calibration_utils import (
    BinnedCalibrator, feature_columns, hold_result, is_buy, signals_from, time_split,
)


class ConformalLGBMStrategy(Strategy):
    """LightGBM with class-conditional (Mondrian) inductive conformal
    prediction — the model abstains instead of guessing.

    A later slice of the window is held out as a calibration set. For each
    class, nonconformity = 1 - p(class); a test row's p-value for a class is
    the share of calibration rows of that class at least as nonconforming.
    A Buy is only allowed when Buy is plausible (p_Buy > alpha) and both
    Hold and Sell are rejected (p <= alpha) — a singleton {Buy} prediction
    set. Confidence is then the binned-calibrated P(Buy) on the same
    holdout, so it reads as a hit rate. See Kaya & Nguyen (PMLR 2025),
    "Conformal Prediction for Reliable Stock Selections".
    """
    data_source = "features"
    handles_nan = True

    def __init__(
        self,
        alpha: float = 0.3,
        calib_fraction: float = 0.25,
        buy_threshold: float = 0.5,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.alpha = alpha
        self.calib_fraction = calib_fraction
        self.buy_threshold = buy_threshold
        self._model = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=200, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=random_state,
            n_jobs=4, verbose=-1,
        )
        self._calib = BinnedCalibrator()
        self._calib_scores: dict[str, np.ndarray] = {}
        self._classes: list[str] = []
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = feature_columns(df)
        train, calib = time_split(df, self.calib_fraction)
        self._calib_scores = {}
        if len(train) == 0 or len(calib) == 0:
            return
        self._model.fit(train[self._feature_cols].to_numpy(), train["label"].to_numpy())
        self._classes = [str(c) for c in self._model.classes_]
        if "Buy" not in self._classes:
            return

        proba = self._model.predict_proba(calib[self._feature_cols].to_numpy())
        y = calib["label"].to_numpy()
        for i, c in enumerate(self._classes):
            self._calib_scores[c] = np.sort(1.0 - proba[y == c, i])
        self._calib.fit(proba[:, self._classes.index("Buy")], is_buy(calib))

    def _p_values(self, proba: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for i, c in enumerate(self._classes):
            scores = self._calib_scores.get(c)
            if scores is None or len(scores) == 0:
                out[c] = np.ones(len(proba))
                continue
            test = 1.0 - proba[:, i]
            n_ge = len(scores) - np.searchsorted(scores, test, side="left")
            out[c] = (n_ge + 1) / (len(scores) + 1)
        return out

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        if not self._calib_scores:
            return hold_result(len(df))
        proba = self._model.predict_proba(df[self._feature_cols].to_numpy())
        pv = self._p_values(proba)
        singleton_buy = pv["Buy"] > self.alpha
        for c in self._classes:
            if c != "Buy":
                singleton_buy &= pv[c] <= self.alpha
        calibrated = self._calib.predict(proba[:, self._classes.index("Buy")])
        return signals_from(np.where(singleton_buy, calibrated, 0.0), self.buy_threshold)
