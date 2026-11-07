"""Shared helpers for the calibrated strategies (meta-labeling, conformal,
ranker, regime-gated consensus).

These strategies report confidence as a *calibrated hit probability* —
the empirical chance a Buy's 5-day forward return clears the label
threshold — not a raw model score. That is what makes "0.6" mean "right
about 60% of the time", the property the over-confidence trap breaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_META = {"time", "ticker", "label", "forward_return_5d"}


class BinnedCalibrator:
    """Monotone score -> hit-rate map built from bins of >= min_bin samples.

    Plain isotonic regression lets a handful of extreme calibration rows set
    the top of the curve (backtest: a "0.9-1.0" bucket that hit 5%). Here
    every point on the curve is the mean of at least min_bin real outcomes,
    then made monotone with isotonic regression over the bins, and scores
    outside the seen range clamp to the end bins.
    """

    def __init__(self, min_bin: int = 200) -> None:
        self.min_bin = min_bin
        self._x = np.array([0.0])
        self._y = np.array([0.0])

    def fit(self, scores: np.ndarray, hits: np.ndarray) -> "BinnedCalibrator":
        from sklearn.isotonic import IsotonicRegression
        order = np.argsort(scores, kind="stable")
        s, h = np.asarray(scores)[order], np.asarray(hits, dtype=float)[order]
        if len(s) == 0:
            return self
        n_bins = max(1, len(s) // self.min_bin)
        chunks = np.array_split(np.arange(len(s)), n_bins)
        x = np.array([s[c].mean() for c in chunks])
        y = np.array([h[c].mean() for c in chunks])
        w = np.array([len(c) for c in chunks], dtype=float)
        self._x = x
        self._y = IsotonicRegression(y_min=0.0, y_max=1.0).fit(x, y, sample_weight=w).predict(x)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(scores, self._x, self._y)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


def time_split(df: pd.DataFrame, holdout_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on calendar time so the holdout is strictly later than the fit
    part — a random split would leak same-day cross-ticker information."""
    if "time" not in df.columns:
        cut = int(len(df) * (1 - holdout_fraction))
        return df.iloc[:cut], df.iloc[cut:]
    times = np.sort(df["time"].unique())
    cut_time = times[int(len(times) * (1 - holdout_fraction))]
    return df[df["time"] < cut_time], df[df["time"] >= cut_time]


def is_buy(df: pd.DataFrame) -> np.ndarray:
    return (df["label"].to_numpy() == "Buy").astype(int)


def hold_result(n: int):
    from src.strategies.base import PredictionResult
    return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))


def signals_from(confidence: np.ndarray, buy_threshold: float):
    from src.strategies.base import PredictionResult
    conf = pd.Series(np.clip(confidence, 0.0, 1.0))
    signal = pd.Series(np.where(conf >= buy_threshold, "Buy", "Hold"))
    return PredictionResult(confidence=conf, signal=signal)
