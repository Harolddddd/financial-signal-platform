from datetime import date, timedelta

import polars as pl
import pytest

from src.league.scoring import score_predictions


def _frames(scores, rets):
    d0 = date(2026, 1, 5)
    times = [d0 + timedelta(days=i) for i in range(len(scores))]
    preds = pl.DataFrame({"ticker": ["A"] * len(scores), "time": times, "buy_score": scores})
    outs = pl.DataFrame({"ticker": ["A"] * len(rets), "time": times, "forward_return_5d": rets})
    return preds, outs


def test_cost_reduces_return():
    p, o = _frames([1.0] * 3, [0.01, 0.02, 0.0])
    r = score_predictions(p, o, cost_bps=10)
    assert r["mean_net_return"] == pytest.approx(0.01 - 0.002)
    assert r["n_trades"] == 3
    assert r["precision"] == pytest.approx(2 / 3)


def test_no_trades_scores_zero():
    p, o = _frames([0.0] * 5, [0.01] * 5)
    r = score_predictions(p, o, cost_bps=10)
    assert r["net_sharpe"] == 0.0 and r["n_trades"] == 0 and r["precision"] == 0.0


def test_sharpe_annualized_over_trades():
    p, o = _frames([0.7] * 4, [0.012, 0.002, 0.012, 0.002])
    r = score_predictions(p, o, cost_bps=0)
    import math
    assert r["net_sharpe"] == pytest.approx(0.007 / pl.Series([0.012, 0.002, 0.012, 0.002]).std() * math.sqrt(252 / 5))
