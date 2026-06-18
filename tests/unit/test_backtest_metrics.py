import pytest
from src.backtesting.metrics import Trade, BacktestMetrics, compute_metrics


def _trade(return_pct: float, actual: str = "Buy", predicted: str = "Buy",
           confidence: float = 0.8) -> Trade:
    entry = 100.0
    return Trade(
        entry_date="2024-01-02",
        exit_date="2024-01-09",
        entry_price=entry,
        exit_price=entry * (1 + return_pct),
        predicted_label=predicted,
        actual_label=actual,
        return_pct=return_pct,
        confidence=confidence,
    )


def test_empty_trades_returns_zero_metrics():
    m = compute_metrics([])
    assert m.n_trades == 0
    assert m.win_rate == 0.0
    assert m.sharpe_ratio == 0.0


def test_all_winning_trades():
    trades = [_trade(0.03), _trade(0.05), _trade(0.02)]
    m = compute_metrics(trades)
    assert m.win_rate == 1.0
    assert m.n_trades == 3
    assert m.total_return_pct > 0
    assert m.profit_factor == float("inf") or m.profit_factor > 0


def test_all_losing_trades():
    trades = [_trade(-0.03), _trade(-0.05)]
    m = compute_metrics(trades)
    assert m.win_rate == 0.0
    assert m.total_return_pct < 0


def test_max_drawdown_is_non_negative():
    trades = [_trade(0.05), _trade(-0.10), _trade(0.03)]
    m = compute_metrics(trades)
    assert m.max_drawdown_pct >= 0.0


def test_precision_buy_correct():
    trades = [
        _trade(0.03, actual="Buy",  predicted="Buy"),
        _trade(-0.02, actual="Hold", predicted="Buy"),
        _trade(0.01, actual="Buy",  predicted="Buy"),
    ]
    m = compute_metrics(trades)
    assert abs(m.precision_buy - 2 / 3) < 1e-9


def test_sharpe_ratio_positive_for_consistent_gains():
    trades = [_trade(0.02)] * 20
    m = compute_metrics(trades)
    assert m.sharpe_ratio > 0
