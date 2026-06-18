"""
End-to-end smoke test: synthetic data → features → model → backtest → signal → grade.
No DB, no external APIs, no Airflow. Runs entirely in-memory.
"""
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_CLASSES = ["Buy", "Hold", "Sell"]
_FEATURE_COLS = [f"f{i}" for i in range(10)]


def _make_full_df(n: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(99)
    base = datetime(2022, 1, 3, tzinfo=timezone.utc)
    labels = [_CLASSES[i % 3] for i in range(n)]
    closes = [100.0 + i * 0.1 for i in range(n)]
    returns = [0.03 if l == "Buy" else -0.01 if l == "Sell" else 0.005 for l in labels]
    features = {f"f{j}": rng.standard_normal(n).tolist() for j in range(10)}
    return pl.DataFrame({
        "time":              [base + timedelta(days=i) for i in range(n)],
        "ticker":            ["AAPL"] * n,
        "close":             closes,
        "forward_return_5d": returns,
        "label":             labels,
        **features,
    })


def test_full_pipeline_smoke():
    from src.models.zoo.random_forest import RandomForestClassifier_
    from src.models.evaluator import evaluate
    from src.backtesting.engine import run_backtest
    from src.backtesting.walk_forward import walk_forward_backtest
    from src.backtesting.grader import grade_model
    from src.signals.signal_engine import generate_signals
    from src.models.registry import save_model, load_model, list_models
    from src.models.base_classifier import EvaluationResult

    df = _make_full_df()

    # 1. Train model
    X = df.select(_FEATURE_COLS).to_numpy()
    y = df["label"].to_numpy()
    clf = RandomForestClassifier_(n_estimators=20)
    clf.fit(X[:300], y[:300])

    # 2. Evaluate
    y_pred = clf.predict(X[300:])
    eval_result = evaluate(y[300:], y_pred)
    assert 0 <= eval_result.precision_buy <= 1

    # 3. Backtest
    bt = run_backtest(clf, df[300:], _FEATURE_COLS, confidence_threshold=0.0)
    assert isinstance(bt.metrics.sharpe_ratio, float)

    # 4. Walk-forward backtest
    wf = walk_forward_backtest(
        df, clf, _FEATURE_COLS,
        train_window_days=150, test_window_days=20, step_days=20,
    )
    assert len(wf.folds) >= 1

    # 5. Grade
    grade = grade_model("random_forest", wf.folds[-1].metrics)
    assert grade.grade.value in {"A", "B", "C", "D"}

    # 6. Registry save/load
    with tempfile.TemporaryDirectory() as tmp:
        reg = Path(tmp)
        path = save_model(clf, eval_result, clf.default_params, _FEATURE_COLS, reg)
        assert path.exists()
        loaded = load_model("random_forest", reg)
        assert loaded.predict(X[:1]).shape == (1,)

    # 7. Signal engine
    signals = generate_signals(clf, df, _FEATURE_COLS, confidence_threshold=0.0)
    for s in signals:
        assert s.label == "Buy"
        assert 0.0 <= s.buy_probability <= 1.0

    print(f"Smoke test passed: grade={grade.grade.value}, "
          f"precision={eval_result.precision_buy:.3f}, "
          f"signals={len(signals)}, folds={len(wf.folds)}")
