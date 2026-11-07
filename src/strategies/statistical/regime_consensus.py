from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.strategies.base import Strategy, PredictionResult
from src.strategies.statistical.calibration_utils import (
    BinnedCalibrator, feature_columns, hold_result, is_buy, signals_from, time_split,
)

_REGIME_COLS = ["vix_level", "hist_vol_21", "rel_strength_spy"]


class RegimeConsensusStrategy(Strategy):
    """Diverse-model consensus with a market-regime gate — aimed at the
    "everyone agreed and still lost" trap.

    1. Three structurally different members (bagged random-split trees,
       boosted trees, scaled logistic regression) are each binned-
       calibrated on a later holdout so their probabilities share a scale.
    2. Consensus = mean calibrated P(Buy) minus their spread: disagreement
       lowers confidence instead of being averaged away.
    3. A Gaussian mixture clusters days by volatility / VIX / relative
       strength. Regimes where the consensus' holdout hit rate does not
       beat that regime's own Buy base rate are blocked entirely.
    Confidence = binned calibration of consensus -> holdout hit rate.
    """
    data_source = "features"

    def __init__(
        self,
        holdout_fraction: float = 0.3,
        buy_threshold: float = 0.5,
        n_regimes: int = 3,
        candidate_quantile: float = 0.9,
        min_regime_candidates: int = 30,
        random_state: int = 42,
    ) -> None:
        self.holdout_fraction = holdout_fraction
        self.buy_threshold = buy_threshold
        self.n_regimes = n_regimes
        self.candidate_quantile = candidate_quantile
        self.min_regime_candidates = min_regime_candidates
        self.random_state = random_state
        self._members = [
            ExtraTreesClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                n_jobs=4, random_state=random_state,
            ),
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=random_state,
            ),
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
        ]
        self._member_calib: list[BinnedCalibrator] = []
        self._consensus_calib = BinnedCalibrator()
        self._regime_model: GaussianMixture | None = None
        self._regime_cols: list[str] = []
        self._allowed_regimes: set[int] = set()
        self._fitted = False
        self._feature_cols: list[str] = []

    def _member_buy_proba(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for m in self._members:
            classes = list(m.classes_)
            cols.append(m.predict_proba(X)[:, classes.index("Buy")] if "Buy" in classes else np.zeros(len(X)))
        return np.column_stack(cols)

    def _consensus(self, X: np.ndarray) -> np.ndarray:
        raw = self._member_buy_proba(X)
        cal = np.column_stack([calib.predict(raw[:, i]) for i, calib in enumerate(self._member_calib)])
        return cal.mean(axis=1) - cal.std(axis=1)

    def _regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self._regime_model is None:
            return np.zeros(len(df), dtype=int)
        return self._regime_model.predict(df[self._regime_cols].fillna(0.0).to_numpy())

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = feature_columns(df)
        self._fitted = False
        train, hold = time_split(df, self.holdout_fraction)
        if len(train) == 0 or len(hold) == 0:
            return
        X_train = train[self._feature_cols].fillna(0.0).to_numpy()
        y_train = train["label"].to_numpy()
        for m in self._members:
            m.fit(X_train, y_train)

        X_hold = hold[self._feature_cols].fillna(0.0).to_numpy()
        y_hold = is_buy(hold)
        raw = self._member_buy_proba(X_hold)
        self._member_calib = []
        for i in range(raw.shape[1]):
            calib = BinnedCalibrator()
            calib.fit(raw[:, i], y_hold)
            self._member_calib.append(calib)
        consensus = self._consensus(X_hold)
        self._consensus_calib.fit(consensus, y_hold)

        self._regime_cols = [c for c in _REGIME_COLS if c in df.columns]
        self._regime_model = None
        if self._regime_cols:
            self._regime_model = GaussianMixture(
                n_components=self.n_regimes, covariance_type="diag",
                random_state=self.random_state,
            ).fit(train[self._regime_cols].fillna(0.0).to_numpy())
        regimes = self._regimes(hold)
        cutoff = np.quantile(consensus, self.candidate_quantile)
        candidate = consensus >= cutoff
        self._allowed_regimes = set()
        for r in np.unique(regimes):
            in_r = regimes == r
            picked = candidate & in_r
            if picked.sum() < self.min_regime_candidates:
                continue
            if y_hold[picked].mean() > y_hold[in_r].mean():
                self._allowed_regimes.add(int(r))
        self._fitted = True

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        if not self._fitted:
            return hold_result(len(df))
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        confidence = self._consensus_calib.predict(self._consensus(X))
        allowed = np.isin(self._regimes(df), list(self._allowed_regimes))
        return signals_from(np.where(allowed, confidence, 0.0), self.buy_threshold)
