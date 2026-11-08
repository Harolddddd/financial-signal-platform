from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from src.strategies.threads import N_JOBS
from src.strategies.base import Strategy, PredictionResult
from src.strategies.statistical.calibration_utils import (
    BinnedCalibrator, feature_columns, hold_result, is_buy, signals_from, time_split,
)


class CrossSectionalRankerStrategy(Strategy):
    """LightGBM LambdaRank learning-to-rank across stocks on each day
    (Qlib-style cross-sectional selection; cf. Gu, Kelly & Xiu 2020).

    Instead of asking "will this stock go up?" it learns "which stocks
    beat their peers on the same day", which removes the market-wide move
    that makes every model agree in a rally and lose together in a selloff.
    Relevance per day = quintile of forward_return_5d. predict() sees one
    ticker at a time, so a score only becomes a Buy when binned
    calibration on a later holdout maps it to a hit probability >= the
    threshold.
    """
    data_source = "features"
    handles_nan = True

    def __init__(
        self,
        calib_fraction: float = 0.25,
        buy_threshold: float = 0.5,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_group_size: int = 5,
        random_state: int = 42,
        use_news_macro: bool = False,
        market: str = "us",
    ) -> None:
        from src.features.news_macro_features import overlay_cols

        # With news/macro: _select_cols passes these columns through and
        # feature_columns() picks them up like any other feature.
        self.market = market
        self.extra_feature_cols: tuple[str, ...] = tuple(overlay_cols(market)) if use_news_macro else ()
        self.calib_fraction = calib_fraction
        self.buy_threshold = buy_threshold
        self.min_group_size = min_group_size
        self._model = LGBMRanker(
            objective="lambdarank", n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=random_state, n_jobs=N_JOBS, verbose=-1,
        )
        self._calib = BinnedCalibrator()
        self._fitted = False
        self._feature_cols: list[str] = []

    def _ranking_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["forward_return_5d"]).sort_values("time", kind="stable")
        sizes = df.groupby("time")["forward_return_5d"].transform("size")
        df = df[sizes >= self.min_group_size]
        pct = df.groupby("time")["forward_return_5d"].rank(pct=True)
        return df.assign(_rel=np.minimum((pct * 5).astype(int), 4))

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = feature_columns(df)
        self._fitted = False
        if "time" not in df.columns or "forward_return_5d" not in df.columns:
            return
        train, calib = time_split(df, self.calib_fraction)
        train = self._ranking_frame(train)
        if len(train) == 0 or len(calib) == 0:
            return
        groups = train.groupby("time", sort=False).size().to_numpy()
        self._model.fit(train[self._feature_cols].to_numpy(), train["_rel"].to_numpy(), group=groups)
        scores = self._model.predict(calib[self._feature_cols].to_numpy())
        self._calib.fit(scores, is_buy(calib))
        self._fitted = True

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        if not self._fitted:
            return hold_result(len(df))
        scores = self._model.predict(df[self._feature_cols].to_numpy())
        return signals_from(self._calib.predict(scores), self.buy_threshold)
