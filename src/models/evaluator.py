import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.models.base_classifier import EvaluationResult

_CLASSES = ["Buy", "Hold", "Sell"]


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> EvaluationResult:
    labels = sorted(set(y_true) | set(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    prec_buy = float(precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )[labels.index("Buy")] if "Buy" in labels else 0.0)
    rec_buy = float(recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )[labels.index("Buy")] if "Buy" in labels else 0.0)
    f1_buy = float(f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )[labels.index("Buy")] if "Buy" in labels else 0.0)
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return EvaluationResult(
        accuracy=accuracy,
        precision_buy=prec_buy,
        recall_buy=rec_buy,
        f1_buy=f1_buy,
        f1_macro=f1_macro,
        confusion_matrix=cm,
        class_labels=labels,
        n_samples=len(y_true),
    )
