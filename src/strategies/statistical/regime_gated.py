"""Two LightGBM models, one per volatility regime (calm vs stressed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import PredictionResult
from src.strategies.statistical.pooled_lgbm import PooledLGBMStrategy

_VOL_COL = "vix_level"  # VIX for US; synthetic realized-vol index for China


class RegimeGatedStrategy(PooledLGBMStrategy):
    """Splits training rows at the median of the volatility index seen in
    training and fits one model per side; predict routes each row to the model
    of its regime. Falls back to a single model if a side is too small."""

    def __init__(self, market: str = "us", min_rows: int = 500, **kw) -> None:
        super().__init__(use_news_macro=True, market=market, **kw)
        self.min_rows = min_rows
        self.kw = kw
        self._threshold = np.nan
        self._models: dict[str, PooledLGBMStrategy] = {}

    def fit(self, df: pd.DataFrame) -> None:
        self._models = {}
        if _VOL_COL not in df.columns:
            super().fit(df)
            return
        self._threshold = float(df[_VOL_COL].median())
        hi = df[_VOL_COL] > self._threshold
        for name, part in (("high", df[hi]), ("low", df[~hi])):
            if len(part) >= self.min_rows and part["label"].nunique() > 1:
                m = PooledLGBMStrategy(use_news_macro=True, market=self.market, **self.kw)
                m.fit(part)
                self._models[name] = m
        super().fit(df)  # fallback for a missing side

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        base = super().predict(df)
        if not self._models or _VOL_COL not in df.columns:
            return base
        conf = base.confidence.to_numpy(dtype=float).copy()
        sig = base.signal.to_numpy(dtype=object).copy()
        hi = (df[_VOL_COL] > self._threshold).to_numpy()
        for name, mask in (("high", hi), ("low", ~hi)):
            if name in self._models and mask.any():
                r = self._models[name].predict(df[mask].reset_index(drop=True))
                conf[mask] = r.confidence.to_numpy(dtype=float)
                sig[mask] = r.signal.to_numpy(dtype=object)
        return PredictionResult(confidence=pd.Series(conf), signal=pd.Series(sig))
