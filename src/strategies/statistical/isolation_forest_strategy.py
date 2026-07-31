from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class IsolationForestAnomaly(Strategy):
    """Unsupervised anomaly detector: flags statistically unusual multi-feature
    states (fit ignores the label), and buys only when an anomalous state
    coincides with an oversold RSI — a dislocation-driven mean-reversion
    signal, not a directional forecast like the supervised classifiers."""
    data_source = "features"

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05,
                 rsi_oversold: float = 40.0, random_state: int = 42) -> None:
        self._model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            n_jobs=4, random_state=random_state,
        )
        self._feature_cols: list[str] = []
        self.rsi_oversold = rsi_oversold

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        self._model.fit(X)  # unsupervised — label is not used

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        raw_score = self._model.score_samples(X)  # higher = more normal
        neg = -raw_score
        span = np.ptp(neg)
        anomaly = (neg - neg.min()) / (span + 1e-9)

        rsi = df["rsi_14"].fillna(50.0) if "rsi_14" in df.columns else pd.Series(50.0, index=df.index)
        oversold = (rsi < self.rsi_oversold).to_numpy()

        confidence = pd.Series(np.where(oversold, anomaly, 0.0))
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence.reset_index(drop=True),
                                 signal=signal.reset_index(drop=True))
