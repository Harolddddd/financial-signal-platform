import numpy as np
import pandas as pd

from src.paper.exit_model import ExitModel


def _df(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    label = np.where(x + 0.3 * noise > 1.2, "Sell", np.where(x < -1, "Buy", "Hold"))
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "ticker": [f"T{i % 50}" for i in range(n)],
        "rsi_14": x, "macd": rng.standard_normal(n), "label": label,
        "forward_return_5d": rng.standard_normal(n) * 0.02,
    })


def test_flags_the_risky_rows():
    m = ExitModel(feature_cols=["rsi_14", "macd"], top_frac=0.10)
    m.fit(_df())
    assert 0 < m.threshold < 1
    assert m.holdout_precision > m.base_rate * 2      # far better than the base rate of Sell
    live = pd.DataFrame({"ticker": ["RISKY", "SAFE"], "rsi_14": [2.5, -2.0], "macd": [0.0, 0.0]})
    assert m.high_risk(live) == {"RISKY"}


def test_unfitted_or_degenerate_flags_nothing():
    m = ExitModel(feature_cols=["rsi_14", "macd"])
    live = pd.DataFrame({"ticker": ["A"], "rsi_14": [3.0], "macd": [0.0]})
    assert m.high_risk(live) == set()
    d = _df()
    d["label"] = "Hold"                                 # no Sell examples at all
    m.fit(d)
    assert m.high_risk(live) == set()


def test_missing_feature_columns_tolerated():
    m = ExitModel(feature_cols=["rsi_14", "macd", "news_net_z60"])
    m.fit(_df())
    live = pd.DataFrame({"ticker": ["RISKY"], "rsi_14": [2.5], "macd": [0.0]})
    assert m.high_risk(live) == {"RISKY"}
