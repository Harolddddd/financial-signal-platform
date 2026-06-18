from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _make_feature_df(n: int = 500) -> pl.DataFrame:
    from dashboard.config import FEATURE_COLS
    rng = np.random.default_rng(0)
    data: dict = {col: rng.random(n).tolist() for col in FEATURE_COLS}
    data["time"]   = [datetime(2005, 1, 1, tzinfo=timezone.utc)] * n
    data["ticker"] = ["AAPL"] * n
    data["label"]  = rng.choice(["Buy", "Hold", "Sell"], n).tolist()
    return pl.DataFrame(data)


def test_train_and_save_creates_registry_files(tmp_path):
    from scripts.train_models import train_and_save
    from src.models.zoo.random_forest import RandomForestClassifier_
    from dashboard.config import FEATURE_COLS

    df = _make_feature_df()
    model = RandomForestClassifier_()
    path = train_and_save(model, df, FEATURE_COLS, tmp_path)

    assert path.exists()
    json_path = path.with_suffix(".json")
    assert json_path.exists()


def test_train_and_save_writes_valid_json(tmp_path):
    import json
    from scripts.train_models import train_and_save
    from src.models.zoo.random_forest import RandomForestClassifier_
    from dashboard.config import FEATURE_COLS

    df = _make_feature_df()
    model = RandomForestClassifier_()
    path = train_and_save(model, df, FEATURE_COLS, tmp_path)

    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["model_name"] == "random_forest"
    assert "accuracy" in meta["evaluation"]
    assert meta["feature_cols"] == FEATURE_COLS


def test_train_and_save_raises_on_empty_df(tmp_path):
    from scripts.train_models import train_and_save
    from src.models.zoo.random_forest import RandomForestClassifier_
    from dashboard.config import FEATURE_COLS

    df = pl.DataFrame({col: [] for col in FEATURE_COLS + ["time", "ticker", "label"]})
    with pytest.raises(ValueError, match="No training data"):
        train_and_save(RandomForestClassifier_(), df, FEATURE_COLS, tmp_path)
