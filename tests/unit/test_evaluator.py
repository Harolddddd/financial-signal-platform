import numpy as np
import pytest
from src.models.evaluator import evaluate
from src.models.base_classifier import EvaluationResult


def test_evaluate_perfect_predictions():
    y = np.array(["Buy", "Hold", "Sell", "Buy", "Buy"])
    result = evaluate(y, y)
    assert result.accuracy == 1.0
    assert result.precision_buy == 1.0
    assert result.recall_buy == 1.0
    assert result.f1_buy == 1.0


def test_evaluate_returns_evaluation_result():
    y_true = np.array(["Buy", "Hold", "Sell", "Buy", "Hold"])
    y_pred = np.array(["Buy", "Sell", "Sell", "Hold", "Hold"])
    result = evaluate(y_true, y_pred)
    assert isinstance(result, EvaluationResult)
    assert result.n_samples == 5


def test_confusion_matrix_shape():
    y_true = np.array(["Buy", "Hold", "Sell"] * 10)
    y_pred = np.array(["Buy", "Hold", "Buy"] * 10)
    result = evaluate(y_true, y_pred)
    assert len(result.confusion_matrix) == 3
    assert all(len(row) == 3 for row in result.confusion_matrix)


def test_class_labels_sorted():
    y_true = np.array(["Sell", "Buy", "Hold"])
    y_pred = np.array(["Sell", "Buy", "Hold"])
    result = evaluate(y_true, y_pred)
    assert result.class_labels == ["Buy", "Hold", "Sell"]


def test_all_hold_predictions_buy_precision_zero():
    y_true = np.array(["Buy", "Buy", "Hold", "Hold"])
    y_pred = np.array(["Hold", "Hold", "Hold", "Hold"])
    result = evaluate(y_true, y_pred)
    assert result.precision_buy == 0.0
    assert result.recall_buy == 0.0
