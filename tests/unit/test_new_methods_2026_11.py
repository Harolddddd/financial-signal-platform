"""Methods added 2026-11-10: short-term reversal, 52-week-high momentum, DoubleEnsemble."""
import numpy as np
import pandas as pd

from src.strategies.registry import list_strategies, load_strategy


def _prices(path):
    n = len(path)
    close = pd.Series(path, dtype=float)
    return pd.DataFrame({"time": pd.date_range("2022-01-01", periods=n), "open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close, "volume": 1e6})


def test_new_methods_registered_for_both_markets():
    for name in ("short_term_reversal", "high52_momentum", "double_ensemble"):
        assert name in list_strategies("us") and name in list_strategies("china")


def test_reversal_buys_sharp_drop_in_uptrend_only():
    rng = np.random.default_rng(0)
    up = 100 * np.cumprod(1 + 0.003 + rng.normal(0, 0.006, 300))
    path = np.concatenate([up, up[-1] * np.array([0.985, 0.97, 0.955, 0.94, 0.925])])
    r = load_strategy("short_term_reversal").predict(_prices(path))
    assert r.signal.iloc[-1] == "Buy" and r.confidence.iloc[-1] >= 0.6
    assert (r.signal.iloc[:250] == "Buy").mean() < 0.05  # quiet uptrend: Buys are rare
    down = 100 * np.cumprod(1 - 0.003 + rng.normal(0, 0.01, 300))
    path = np.concatenate([down, down[-1] * np.array([0.97, 0.94, 0.91, 0.88, 0.86])])
    assert load_strategy("short_term_reversal").predict(_prices(path)).signal.iloc[-1] == "Hold"  # falling knife


def test_high52_buys_near_high_not_far_below():
    rng = np.random.default_rng(1)
    rise = 100 * np.cumprod(1 + 0.002 + rng.normal(0, 0.005, 300))
    dip = rise[-1] * np.linspace(0.99, 0.9, 30)          # pull back ~10% for a month
    back = dip[-1] * np.cumprod(np.full(40, 1.006))     # then climb back to the high
    path = np.concatenate([rise, dip, back])
    r = load_strategy("high52_momentum").predict(_prices(path))
    buys = np.where(r.signal == "Buy")[0]
    assert len(buys) and buys.max() >= 330                # the return to the high is a Buy
    assert (r.signal.iloc[330:] == "Buy").sum() == 1      # once, not every day near the high
    crash = np.concatenate([path, path[-1] * np.linspace(0.99, 0.7, 60)])
    r = load_strategy("high52_momentum").predict(_prices(crash))
    assert r.signal.iloc[-1] == "Hold"
    assert len(r.signal) == len(crash) and len(r.confidence) == len(crash)


def _feature_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    score = X[:, 0] + 0.5 * X[:, 1]
    label = np.where(score > 0.8, "Buy", np.where(score < -0.8, "Sell", "Hold"))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["label"], df["ticker"], df["forward_return_5d"] = label, "T", score / 100
    df["time"] = pd.Timestamp("2024-01-01")
    return df


def test_double_ensemble_learns_and_is_deterministic():
    train, test = _feature_frame(seed=0), _feature_frame(seed=1)
    a = load_strategy("double_ensemble")
    a.fit(train)
    ra = a.predict(test)
    assert len(ra.confidence) == len(test)
    is_buy = test["label"] == "Buy"
    assert ra.confidence[is_buy.values].mean() > ra.confidence[~is_buy.values].mean() + 0.2
    assert a._models and len(a._models[-1][1]) < len(a._models[0][1])  # later sub-models drop features
    b = load_strategy("double_ensemble")
    b.fit(train)
    assert np.allclose(ra.confidence, b.predict(test).confidence)
