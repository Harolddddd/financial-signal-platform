"""Cost-aware score of a method from its walk-forward prediction log."""
from __future__ import annotations

import math

import polars as pl

_HOLD_DAYS = 5  # labels and forward returns are 5-day


def score_predictions(preds: pl.DataFrame, outcomes: pl.DataFrame, cost_bps: float) -> dict:
    """preds: ticker,time,buy_score. outcomes: ticker,time,forward_return_5d.

    Every row with buy_score > 0 is a trade held 5 days; its net return is the
    forward return minus a round trip of costs (2 * cost_bps).
    """
    trades = (
        preds.filter(pl.col("buy_score") > 0)
        .join(outcomes.select("ticker", "time", "forward_return_5d"), on=["ticker", "time"], how="inner")
        .drop_nulls("forward_return_5d")
    )
    net = trades["forward_return_5d"] - 2 * cost_bps / 1e4
    n = len(net)
    if n == 0:
        return {"net_sharpe": 0.0, "n_trades": 0, "precision": 0.0, "mean_net_return": 0.0}
    mean = float(net.mean())
    std = float(net.std()) if n >= 2 else 0.0
    sharpe = mean / std * math.sqrt(252 / _HOLD_DAYS) if n >= 2 and std > 0 else 0.0
    return {
        "net_sharpe": sharpe,
        "n_trades": n,
        "precision": float((net > 0).mean()),
        "mean_net_return": mean,
    }
