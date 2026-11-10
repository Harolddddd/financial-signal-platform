from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from src.backtesting.strategy_runner import walk_forward_backtest_strategy
from src.strategies.base import PredictionResult, Strategy


def _df(n_tickers=6, days=400):
    rng = np.random.default_rng(0)
    rows = []
    t0 = datetime(2024, 1, 1)
    for k in range(n_tickers):
        for d in range(days):
            f = float(rng.normal())
            rows.append({"time": t0 + timedelta(days=d), "ticker": f"T{k}", "close": 10.0 + k, "f1": f,
                         "label": "Buy" if f > 0 else "Hold", "forward_return_5d": f / 100})
    return pl.DataFrame(rows)


class _RowWise(Strategy):
    """Pure row-wise model: confidence depends on the row only."""
    data_source = "features"

    def __init__(self):
        self.calls = 0

    def fit(self, df):
        self.w = float(df["f1"].mean())

    def predict(self, df):
        self.calls += 1
        c = 1 / (1 + np.exp(-(df["f1"].to_numpy() - self.w)))
        return PredictionResult(confidence=pd.Series(c), signal=pd.Series(["Buy" if x >= 0.6 else "Hold" for x in c]))


class _CrossSection(_RowWise):
    """Depends on the whole frame (rank within what it is given): must not be batched."""

    def predict(self, df):
        self.calls += 1
        c = df["f1"].rank(pct=True).to_numpy()
        return PredictionResult(confidence=pd.Series(c), signal=pd.Series(["Buy" if x >= 0.6 else "Hold" for x in c]))


def _run(strategy, df):
    log = []
    wf = walk_forward_backtest_strategy(df, strategy, ["close"], ["f1"], train_window_days=100,
                                        test_window_days=21, step_days=21, pred_log=log)
    return wf, sorted(log)


def _per_ticker_baseline(strategy_cls, df, monkeypatch):
    from src.backtesting import strategy_runner as sr
    monkeypatch.setattr(sr, "_BATCH_PREDICT", False)
    out = _run(strategy_cls(), df)
    monkeypatch.setattr(sr, "_BATCH_PREDICT", True)
    return out


def test_row_wise_model_predicts_once_per_fold_with_same_results(monkeypatch):
    df = _df()
    base_wf, base_log = _per_ticker_baseline(_RowWise, df, monkeypatch)
    s = _RowWise()
    wf, log = _run(s, df)
    n_folds, n_tickers = len(wf.folds), 6
    # fold 0 checks both ways (6 + 1 calls), every later fold is one batched call
    assert s.calls == n_tickers + 1 + (n_folds - 1)
    assert log == base_log
    assert wf.mean_sharpe == base_wf.mean_sharpe


def test_frame_dependent_model_keeps_per_ticker_predict(monkeypatch):
    df = _df()
    base_wf, base_log = _per_ticker_baseline(_CrossSection, df, monkeypatch)
    s = _CrossSection()
    wf, log = _run(s, df)
    assert log == base_log  # identical to the old behaviour
    assert s.calls == 6 + 1 + 6 * (len(wf.folds) - 1)  # check failed: per ticker from then on


def test_selected_columns_do_not_depend_on_hash_seed():
    """Column order must be the same in every process: tree models with a fixed seed
    still build different trees when the features come in a different order."""
    import os
    import subprocess
    import sys
    code = (
        "import polars as pl\n"
        "from src.backtesting.strategy_runner import _select_cols\n"
        "from src.strategies.base import Strategy\n"
        "class S(Strategy):\n"
        "    data_source = 'features'\n"
        "    def predict(self, df): pass\n"
        "cols = ['time','ticker','close','label','forward_return_5d'] + [f'f{i}' for i in range(30)]\n"
        "df = pl.DataFrame({c: [1] for c in cols})\n"
        "print(','.join(_select_cols(df, S(), ['close'], [f'f{i}' for i in range(30)]).columns))\n"
    )
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[2])
    outs = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": root}
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        outs.add(r.stdout.strip())
    assert len(outs) == 1
