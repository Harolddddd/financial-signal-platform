from __future__ import annotations
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from src.strategies.base import Strategy, PredictionResult

_META = {"time", "ticker", "label", "forward_return_5d"}


class StackingEnsembleStrategy(Strategy):
    """Meta-learner stacking the platform's three strongest base models
    (ExtraTrees, LDA, RandomForest) via a logistic-regression blender.
    StackingClassifier generates each base model's meta-features via
    internal cross-validation, so the blender never sees in-sample
    base-model predictions."""
    data_source = "features"

    def __init__(self, cv: int = 3, random_state: int = 42) -> None:
        base_estimators = [
            ("extra_trees", ExtraTreesClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                n_jobs=-1, random_state=random_state,
            )),
            ("lda", LinearDiscriminantAnalysis(solver="svd")),
            ("random_forest", RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                n_jobs=-1, random_state=random_state,
            )),
        ]
        self._model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=500),
            cv=cv, n_jobs=-1, passthrough=False,
        )
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        self._feature_cols = [c for c in df.columns if c not in _META]
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        y = df["label"].to_numpy()
        self._model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        if not self._feature_cols:
            raise ValueError("Call fit() before predict()")
        X = df[self._feature_cols].fillna(0.0).to_numpy()
        proba = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        if "Buy" not in classes:
            n = len(df)
            return PredictionResult(confidence=pd.Series([0.0] * n), signal=pd.Series(["Hold"] * n))
        buy_idx = classes.index("Buy")
        confidence = pd.Series(proba[:, buy_idx])
        signal = pd.Series(["Buy" if c >= 0.6 else "Hold" for c in confidence])
        return PredictionResult(confidence=confidence, signal=signal)
