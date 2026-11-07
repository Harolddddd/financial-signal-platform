"""News/macro overlay applied on top of strategy signals (US only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.news_macro_features import OVERLAY_COLS
from src.strategies.base import PredictionResult, Strategy


@dataclass(frozen=True)
class OverlayConfig:
    veto_z: float = -1.5
    boost_z: float = 1.5
    min_news: int = 3
    boost_mult: float = 1.10
    dgs2_jump: float = 0.25
    hy_z: float = 2.0
    fomc_window_days: int = 5
    risk_off_mult: float = 0.85
    buy_threshold: float = 0.6


def _col(ctx: pd.DataFrame, name: str, index: pd.Index) -> pd.Series | None:
    if name not in ctx.columns:
        return None
    return pd.to_numeric(ctx[name], errors="coerce").reindex(index)


def _false(index: pd.Index) -> pd.Series:
    return pd.Series(False, index=index)


def apply_overlay(
    signal: pd.Series,
    confidence: pd.Series,
    ctx: pd.DataFrame,
    cfg: OverlayConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = signal.index
    ctx = ctx.reindex(idx) if not ctx.index.equals(idx) else ctx
    was_buy = signal == "Buy"

    def cmp(name, op):
        s = _col(ctx, name, idx)
        return _false(idx) if s is None else op(s).fillna(False).astype(bool)

    # 1. risk-off flags
    f_dgs2 = cmp("macro_dgs2_chg_5d", lambda s: s > cfg.dgs2_jump)
    f_hy = cmp("macro_hy_spread_z20", lambda s: s > cfg.hy_z)
    tone = _col(ctx, "fomc_tone_change", idx)
    days = _col(ctx, "days_since_fomc", idx)
    if tone is None or days is None:
        f_fomc = _false(idx)
    else:
        f_fomc = ((tone > 0) & (days <= cfg.fomc_window_days)).fillna(False)
    n_flags = f_dgs2.astype(int) + f_hy.astype(int) + f_fomc.astype(int)
    conf = confidence.astype(float) * np.power(cfg.risk_off_mult, n_flags)

    # 2/3. news veto and boost
    count = _col(ctx, "news_count_3d", idx)
    z = _col(ctx, "news_net_z60", idx)
    if count is None or z is None:
        veto = boost = _false(idx)
    else:
        enough = count >= cfg.min_news
        veto = (enough & (z < cfg.veto_z)).fillna(False)
        boost = (enough & (z > cfg.boost_z)).fillna(False)
    conf = conf.where(~boost, (conf * cfg.boost_mult).clip(upper=1.0))

    out = signal.copy()
    out[veto & was_buy] = "Hold"
    # 4. low adjusted confidence on a Buy -> Hold
    out[(out == "Buy") & (conf < cfg.buy_threshold)] = "Hold"

    # reasons (Buy rows only)
    parts = [
        ("risk-off: 2y yield jump", f_dgs2),
        ("risk-off: credit spread", f_hy),
        ("risk-off: hawkish FOMC", f_fomc),
        ("BUY suppressed: negative news", veto),
        ("confidence boosted: positive news", boost & ~veto),
    ]
    reason = pd.Series("", index=idx, dtype=object)
    for text, mask in parts:
        m = mask & was_buy
        reason = reason.where(~m, reason.where(reason == "", reason + "; ") + text)
    return out, conf, reason


class OverlayStrategy(Strategy):
    def __init__(self, inner: Strategy, cfg: OverlayConfig = OverlayConfig()):
        self.inner = inner
        self.cfg = cfg
        self.data_source = inner.data_source
        self.last_reason: pd.Series = pd.Series(dtype=object)
        self.last_raw_confidence: pd.Series = pd.Series(dtype=float)

    @staticmethod
    def _strip(df):
        return df.drop(columns=[c for c in OVERLAY_COLS if c in df.columns])

    def fit(self, df: pd.DataFrame) -> None:
        self.inner.fit(self._strip(df))

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        res = self.inner.predict(self._strip(df))
        sig = pd.Series(res.signal.to_numpy(), index=df.index)
        raw = pd.Series(res.confidence.to_numpy(dtype=float), index=df.index)
        ctx = df[[c for c in OVERLAY_COLS if c in df.columns]]
        sig2, conf2, reason = apply_overlay(sig, raw, ctx, self.cfg)
        self.last_raw_confidence = raw
        self.last_reason = reason
        return PredictionResult(confidence=conf2, signal=sig2)
