"""News surprise: sentiment far from its own 60-day norm, on unusually heavy news flow."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.news_macro_features import NEWS_COLS
from src.strategies.statistical.pooled_lgbm import PooledLGBMStrategy


class NewsSurpriseStrategy(PooledLGBMStrategy):
    """Pooled LightGBM on price + news columns plus two surprise terms.

    news_net_z60 is already "today's 3-day sentiment vs its 60-day baseline";
    multiplying by the positive part of news_vol_z20 makes the surprise count
    only when the article flow is abnormally high. Row-wise, so it works on a
    single ticker's test window.
    """

    def __init__(self, market: str = "us", **kw) -> None:
        super().__init__(use_news_macro=True, market=market, **kw)
        self.extra_feature_cols = tuple(NEWS_COLS)

    @staticmethod
    def _with_surprise(df: pd.DataFrame) -> pd.DataFrame:
        z = df["news_net_z60"] if "news_net_z60" in df else pd.Series(0.0, index=df.index)
        v = df["news_vol_z20"] if "news_vol_z20" in df else pd.Series(0.0, index=df.index)
        heavy = np.clip(v.astype(float), 0.0, None)
        return df.assign(news_surprise=z.astype(float) * heavy,
                         news_surprise_abs=np.abs(z.astype(float)) * heavy)

    def fit(self, df: pd.DataFrame) -> None:
        super().fit(self._with_surprise(df))

    def predict(self, df: pd.DataFrame):
        return super().predict(self._with_surprise(df))
