"""Phase B: one LightGBM pooled across tickers on price + news + macro features."""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier

from src.features.news_macro_features import OVERLAY_COLS
from src.strategies.statistical.lightgbm_strategy import LightGBMStrategy
from src.strategies.threads import N_JOBS

_META = {"time", "ticker", "label", "forward_return_5d"}
_OVERLAY = frozenset(OVERLAY_COLS)


class PooledLGBMStrategy(LightGBMStrategy):
    """LightGBM that also learns from the news/macro columns (OVERLAY_COLS).

    `extra_feature_cols` tells `_select_cols` to pass those columns through on
    top of the price features. Early rows have NaN news (history is short);
    LightGBM handles NaN natively. use_news_macro=False gives the price-only
    baseline for comparison.
    """

    def __init__(
        self,
        use_news_macro: bool = True,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_samples: int = 100,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self._model = LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, min_child_samples=min_child_samples,
            class_weight="balanced", random_state=random_state, n_jobs=N_JOBS, verbose=-1,
        )
        self.use_news_macro = use_news_macro
        self.extra_feature_cols: tuple[str, ...] = tuple(OVERLAY_COLS) if use_news_macro else ()

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [
            c for c in df.columns
            if c not in _META and (self.use_news_macro or c not in _OVERLAY)
        ]
        self._model.fit(df[self._feature_cols].to_numpy(), df["label"].to_numpy())
