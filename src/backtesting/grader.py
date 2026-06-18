from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math

from src.backtesting.metrics import BacktestMetrics


class Grade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class ModelGrade:
    model_name: str
    grade: Grade
    composite_score: float
    metrics: BacktestMetrics


def _norm_sharpe(sharpe: float) -> float:
    return float(math.tanh(sharpe / 2))


def _norm_drawdown(drawdown: float) -> float:
    return min(drawdown, 0.50) / 0.50


def grade_model(model_name: str, metrics: BacktestMetrics) -> ModelGrade:
    score = (
        0.40 * metrics.precision_buy
        + 0.30 * _norm_sharpe(metrics.sharpe_ratio)
        + 0.30 * (1.0 - _norm_drawdown(metrics.max_drawdown_pct))
    )
    score = max(0.0, min(1.0, score))

    if score >= 0.65:
        grade = Grade.A
    elif score >= 0.50:
        grade = Grade.B
    elif score >= 0.35:
        grade = Grade.C
    else:
        grade = Grade.D

    return ModelGrade(
        model_name=model_name,
        grade=grade,
        composite_score=round(score, 4),
        metrics=metrics,
    )


def build_leaderboard(grades: list[ModelGrade]) -> list[ModelGrade]:
    return sorted(grades, key=lambda g: g.composite_score, reverse=True)
