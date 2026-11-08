"""Sell model for the paper account: chance a stock lands in the Sell class over the next 5 days.

A LightGBM classifier on price + news + macro features with target label == "Sell"
(5-day forward return below the label threshold). The "high risk" cut is the score
of the riskiest `top_frac` of a later, held-out slice of the training window, so a
flag means "riskier than ~90% of recent stock-days".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.strategies.statistical.calibration_utils import time_split
from src.strategies.threads import N_JOBS


class ExitModel:
    def __init__(self, feature_cols: list[str], top_frac: float = 0.10, holdout_fraction: float = 0.2,
                 random_state: int = 42) -> None:
        self.feature_cols = list(feature_cols)
        self.top_frac = top_frac
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state
        self._model: LGBMClassifier | None = None
        self.threshold = float("nan")
        self.base_rate = 0.0
        self.holdout_precision = 0.0

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        cols = {c: (df[c] if c in df.columns else np.nan) for c in self.feature_cols}
        return pd.DataFrame(cols, index=df.index).astype(float).to_numpy()

    def _new(self) -> LGBMClassifier:
        return LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.05, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, is_unbalance=True,
                              random_state=self.random_state, n_jobs=N_JOBS, verbose=-1)

    def fit(self, df: pd.DataFrame) -> "ExitModel":
        self._model = None
        y_all = (df["label"].to_numpy() == "Sell").astype(int)
        if y_all.min() == y_all.max():
            return self
        train, hold = time_split(df, self.holdout_fraction)
        y_tr = (train["label"].to_numpy() == "Sell").astype(int)
        y_ho = (hold["label"].to_numpy() == "Sell").astype(int)
        if len(hold) == 0 or y_tr.min() == y_tr.max():
            return self
        probe = self._new().fit(self._x(train), y_tr)
        p_ho = probe.predict_proba(self._x(hold))[:, 1]
        self.threshold = float(np.quantile(p_ho, 1 - self.top_frac))
        flagged = p_ho >= self.threshold
        self.base_rate = float(y_ho.mean())
        self.holdout_precision = float(y_ho[flagged].mean()) if flagged.any() else 0.0
        self._model = self._new().fit(self._x(df), y_all)
        return self

    def scores(self, live: pd.DataFrame) -> pd.Series:
        if self._model is None or live.empty:
            return pd.Series(0.0, index=live.index)
        return pd.Series(self._model.predict_proba(self._x(live))[:, 1], index=live.index)

    def high_risk(self, live: pd.DataFrame) -> set[str]:
        if self._model is None or live.empty:
            return set()
        s = self.scores(live)
        return set(live.loc[s >= self.threshold, "ticker"].astype(str))
