import pytest
from src.backtesting.grader import Grade, ModelGrade, grade_model, build_leaderboard
from src.backtesting.metrics import BacktestMetrics


def _metrics(precision: float, sharpe: float, drawdown: float) -> BacktestMetrics:
    return BacktestMetrics(
        n_trades=50, win_rate=0.55, profit_factor=1.5,
        total_return_pct=0.10, sharpe_ratio=sharpe,
        max_drawdown_pct=drawdown, precision_buy=precision,
        recall_buy=0.4, f1_buy=0.48, accuracy=0.55,
    )


def test_high_precision_high_sharpe_low_drawdown_gets_grade_a():
    g = grade_model("rf", _metrics(precision=0.80, sharpe=2.5, drawdown=0.03))
    assert g.grade == Grade.A


def test_low_precision_low_sharpe_high_drawdown_gets_grade_d():
    g = grade_model("nb", _metrics(precision=0.20, sharpe=-1.0, drawdown=0.45))
    assert g.grade == Grade.D


def test_composite_score_between_zero_and_one():
    g = grade_model("rf", _metrics(0.55, 1.0, 0.10))
    assert 0.0 <= g.composite_score <= 1.0


def test_leaderboard_sorted_descending():
    grades = [
        grade_model("m1", _metrics(0.4, 0.5, 0.2)),
        grade_model("m2", _metrics(0.8, 2.0, 0.05)),
        grade_model("m3", _metrics(0.3, 0.1, 0.4)),
    ]
    board = build_leaderboard(grades)
    scores = [g.composite_score for g in board]
    assert scores == sorted(scores, reverse=True)


def test_grade_model_returns_model_grade():
    g = grade_model("xgboost", _metrics(0.6, 1.2, 0.08))
    assert isinstance(g, ModelGrade)
    assert g.model_name == "xgboost"
