from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    predicted_label: str
    actual_label: str
    return_pct: float
    confidence: float


@dataclass
class BacktestMetrics:
    n_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    precision_buy: float
    recall_buy: float
    f1_buy: float
    accuracy: float


def compute_metrics(trades: list[Trade]) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    returns = np.array([t.return_pct for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = float(len(wins) / len(trades))
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_return = float(returns.sum())

    std = returns.std() if len(returns) > 1 else 1e-9
    sharpe = float((returns.mean() / std) * np.sqrt(252 / 5)) if std > 0 else 0.0

    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(abs(drawdown.min()))

    y_true = np.array([t.actual_label for t in trades])
    y_pred = np.array([t.predicted_label for t in trades])
    true_buy = y_true == "Buy"
    pred_buy = y_pred == "Buy"
    tp = int((true_buy & pred_buy).sum())
    fp = int((~true_buy & pred_buy).sum())
    fn = int((true_buy & ~pred_buy).sum())
    tn = int((~true_buy & ~pred_buy).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(trades)

    return BacktestMetrics(
        n_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_return_pct=total_return,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        precision_buy=precision,
        recall_buy=recall,
        f1_buy=f1,
        accuracy=accuracy,
    )
