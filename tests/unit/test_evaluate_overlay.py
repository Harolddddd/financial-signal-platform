from datetime import datetime, timedelta

import pandas as pd
import polars as pl

from scripts import evaluate_overlay
from scripts.evaluate_overlay import evaluate
from src.signals.overlay import OverlayConfig
from src.strategies.base import PredictionResult, Strategy


class _AlwaysBuy(Strategy):
    data_source = "features"

    def fit(self, df):
        pass

    def predict(self, df):
        n = len(df)
        return PredictionResult(
            confidence=pd.Series([0.8] * n, index=df.index),
            signal=pd.Series(["Buy"] * n, index=df.index),
        )


class _Boom(Strategy):
    data_source = "features"

    def fit(self, df):
        raise RuntimeError("boom")

    def predict(self, df):
        raise RuntimeError("boom")


def _df(n_days=120):
    d0 = datetime(2026, 1, 1)
    rows = []
    for tk in ("AAA", "BBB"):
        for i in range(n_days):
            rows.append({
                "ticker": tk, "time": d0 + timedelta(days=i),
                "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0,
                "forward_return_5d": 0.01 if i % 2 else -0.01,
                "label": "Buy" if i % 2 else "Hold",
            })
    return pl.DataFrame(rows)


_SMALL = dict(train_window_days=40, test_window_days=10, step_days=10, min_train_samples=10)


def test_evaluate_returns_plain_and_overlay_rows(monkeypatch):
    monkeypatch.setattr(evaluate_overlay, "load_strategy", lambda name: _AlwaysBuy())
    rows = evaluate(_df(), ["fake"], OverlayConfig(), **_SMALL)
    assert [(r["strategy"], r["variant"]) for r in rows] == [("fake", "plain"), ("fake", "overlay")]
    for r in rows:
        assert set(r) == {"strategy", "variant", "n_trades", "sharpe", "precision_buy"}
        assert r["n_trades"] > 0


def test_evaluate_skips_failing_strategy(monkeypatch):
    monkeypatch.setattr(
        evaluate_overlay, "load_strategy",
        lambda name: _Boom() if name == "bad" else _AlwaysBuy(),
    )
    rows = evaluate(_df(), ["bad", "good"], OverlayConfig(), **_SMALL)
    assert {r["strategy"] for r in rows} == {"good"}
    assert len(rows) == 2


def test_evaluate_pooled_returns_price_only_and_news_macro_rows():
    df = _df().with_columns(
        rsi_14=pl.col("close") + pl.int_range(pl.len()) % 7,
        news_net_3d=pl.lit(None, dtype=pl.Float64),
    )
    rows = evaluate_overlay.evaluate_pooled(df, **_SMALL)
    assert [(r["strategy"], r["variant"]) for r in rows] == [
        ("pooled_lgbm_news", "price_only"), ("pooled_lgbm_news", "news_macro"),
    ]
    for r in rows:
        assert set(r) == {"strategy", "variant", "n_trades", "sharpe", "precision_buy"}
