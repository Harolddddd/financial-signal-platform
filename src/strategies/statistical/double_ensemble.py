from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.strategies.base import PredictionResult, Strategy
from src.strategies.threads import N_JOBS

_META = {"time", "ticker", "label", "forward_return_5d"}


class DoubleEnsembleStrategy(Strategy):
    """A light DoubleEnsemble (Zhang et al., ICDM 2020; a top Qlib baseline on CSI300).

    Trains `n_models` LightGBM sub-models in turn. Each later one
    - re-weights the training rows: rows the ensemble so far gets wrong count more
      (capped, so a few noisy rows cannot take over), and
    - drops the features whose shuffling hurts the ensemble least (the bottom
      `drop_frac`), so sub-models look at the data differently.
    The Buy confidence is the mean Buy probability of the sub-models.
    """

    data_source = "features"
    handles_nan = True

    def __init__(self, n_models: int = 4, n_estimators: int = 60, max_depth: int = 5,
                 learning_rate: float = 0.08, drop_frac: float = 0.2, max_weight: float = 3.0,
                 random_state: int = 42) -> None:
        self.n_models = n_models
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           class_weight="balanced", n_jobs=N_JOBS, verbose=-1)
        self.drop_frac = drop_frac
        self.max_weight = max_weight
        self.random_state = random_state
        self._models: list[tuple[LGBMClassifier, list[int]]] = []
        self._feature_cols: list[str] = []

    @staticmethod
    def _buy_proba(model: LGBMClassifier, X: np.ndarray) -> np.ndarray:
        classes = list(model.classes_)
        if "Buy" not in classes:
            return np.zeros(len(X))
        return model.predict_proba(X)[:, classes.index("Buy")]

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].to_numpy(dtype=float)
        y = df["label"].to_numpy()
        is_buy = (y == "Buy").astype(float)
        rng = np.random.default_rng(self.random_state)
        weights = np.ones(len(y))
        cols = list(range(X.shape[1]))
        ens = np.zeros(len(y))
        self._models = []
        for k in range(self.n_models):
            m = LGBMClassifier(random_state=self.random_state + k, **self.params)
            m.fit(X[:, cols], y, sample_weight=weights)
            self._models.append((m, list(cols)))
            ens = (ens * k + self._buy_proba(m, X[:, cols])) / (k + 1)
            if k == self.n_models - 1:
                break
            # Sample re-weighting: squared error of the ensemble's Buy probability.
            err = (ens - is_buy) ** 2
            weights = np.clip(err / (err.mean() + 1e-12), 1.0 / self.max_weight, self.max_weight)
            # Feature selection: drop the features whose shuffling changes the loss least.
            if len(cols) > 4:
                sample = rng.choice(len(y), size=min(len(y), 5000), replace=False)
                Xs, ys = X[sample][:, cols], is_buy[sample]
                # One predict call for all shuffled copies (one per feature): far less overhead.
                stack = np.tile(Xs, (len(cols) + 1, 1))
                for j in range(len(cols)):
                    rows = slice((j + 1) * len(Xs), (j + 2) * len(Xs))
                    stack[rows, j] = rng.permutation(Xs[:, j])
                err2 = ((self._buy_proba(m, stack) - np.tile(ys, len(cols) + 1)) ** 2).reshape(len(cols) + 1, -1)
                loss = err2.mean(axis=1)
                gain = list(loss[1:] - loss[0])
                n_drop = int(len(cols) * self.drop_frac)
                keep = sorted(np.argsort(gain)[n_drop:])
                cols = [cols[j] for j in keep]

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._models:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].to_numpy(dtype=float)
        conf = np.mean([self._buy_proba(m, X[:, cols]) for m, cols in self._models], axis=0)
        confidence = pd.Series(conf)
        signal = pd.Series(np.where(conf >= 0.6, "Buy", "Hold"))
        return PredictionResult(confidence=confidence, signal=signal)
