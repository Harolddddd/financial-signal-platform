"""Logistic stacker over the current live methods' Buy probabilities."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config.markets import get_market
from src.league.league import live_set
from src.strategies.base import PredictionResult, Strategy
from src.strategies.statistical.calibration_utils import hold_result, is_buy, time_split

logger = logging.getLogger(__name__)


class StackedLiveStrategy(Strategy):
    """Fits each base (live, feature-based) method on the first 70% of the
    training window, learns a logistic stacker on their out-of-sample
    confidences over the last 30%, then refits the bases on the whole window.
    Costs about two base fits per base per fold."""

    data_source = "features"
    handles_nan = True

    def __init__(self, market: str = "us", base: list[str] | None = None,
                 holdout_fraction: float = 0.3, buy_threshold: float = 0.55) -> None:
        self.market = market
        self.base = base
        self.holdout_fraction = holdout_fraction
        self.buy_threshold = buy_threshold
        self._bases: dict[str, Strategy] = {}
        self._stacker: LogisticRegression | None = None
        self.extra_feature_cols: tuple[str, ...] = ()
        names = self.base_names()
        if names:
            from src.strategies.registry import load_strategy
            extra: list[str] = []
            for n in names:
                extra += list(getattr(load_strategy(n), "extra_feature_cols", ()))
            self.extra_feature_cols = tuple(dict.fromkeys(extra))

    def base_names(self) -> list[str]:
        from src.strategies.registry import list_strategies, load_strategy

        cfg = get_market(self.market)
        names = self.base or live_set(cfg.data_root / "cache", self.market) or list(cfg.default_strategies or ())
        known = set(list_strategies(self.market))
        out = []
        for n in names:
            if n.startswith("stacked_live") or n not in known:
                continue
            try:
                if load_strategy(n).data_source == "features":
                    out.append(n)
            except Exception as exc:
                logger.warning("stacked_live: base %s unavailable: %s", n, exc)
        return out

    def _fit_bases(self, df: pd.DataFrame) -> dict[str, Strategy]:
        from src.strategies.registry import load_strategy

        fitted = {}
        for n in self.base_names():
            s = load_strategy(n)
            part = df if getattr(s, "handles_nan", False) else df.dropna()
            if len(part) < 100:
                continue
            s.fit(part)
            fitted[n] = s
        return fitted

    def _matrix(self, bases: dict[str, Strategy], df: pd.DataFrame) -> np.ndarray:
        cols = []
        for s in bases.values():
            try:
                cols.append(s.predict(df.reset_index(drop=True)).confidence.to_numpy(dtype=float))
            except Exception:
                cols.append(np.zeros(len(df)))
        return np.nan_to_num(np.column_stack(cols)) if cols else np.zeros((len(df), 0))

    def fit(self, df: pd.DataFrame) -> None:
        self._stacker, self._bases = None, {}
        train, hold = time_split(df, self.holdout_fraction)
        first = self._fit_bases(train)
        if not first or len(hold) == 0:
            return
        y = is_buy(hold)
        if y.min() == y.max():
            return
        self._stacker = LogisticRegression(max_iter=500).fit(self._matrix(first, hold), y)
        self._bases = self._fit_bases(df)
        if list(self._bases) != list(first):  # a base dropped out on the full window
            self._stacker = None

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if self._stacker is None:
            return hold_result(len(df))
        p = self._stacker.predict_proba(self._matrix(self._bases, df))[:, 1]
        conf = pd.Series(np.clip(p, 0.0, 1.0))
        return PredictionResult(confidence=conf, signal=pd.Series(np.where(conf >= self.buy_threshold, "Buy", "Hold")))
