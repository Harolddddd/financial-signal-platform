"""Incrementally update the trained models using only feature rows added
since each model's last training run — no full retrain.

For each model in the registry:
  1. Load the latest saved version.
  2. Load feature rows with time > that version's trained_at.
  3. Continue training on just those new rows (fit_incremental):
       - random_forest: grow the forest with new trees fit on the new rows
       - xgboost / lightgbm: continue boosting on top of the existing trees
  4. Evaluate on a held-out slice of the new rows and save as the next
     version in the registry.
"""
from pathlib import Path
from datetime import datetime
import logging

from config.markets import get_market
from dashboard.config import FEATURE_COLS, REGISTRY_DIR
from src.features.duckdb_client import load_training_data
from src.models.evaluator import evaluate
from src.models.registry import list_models, load_model, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FEATURE_DIR = get_market("us").data_root / "features"
_TRAIN_RATIO = 0.8
_MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]


def _latest_trained_at(model_name: str, registry_dir: Path) -> datetime | None:
    records = [r for r in list_models(registry_dir) if r.model_name == model_name]
    if not records:
        return None
    latest = max(records, key=lambda r: r.trained_at)
    return datetime.fromisoformat(latest.trained_at)


def incremental_train_one(model_name: str) -> None:
    cutoff = _latest_trained_at(model_name, REGISTRY_DIR)
    if cutoff is None:
        logger.warning("No existing version for %s — skipping (run scripts/train_models.py first)", model_name)
        return

    logger.info("%s — last trained %s, loading rows after that ...", model_name, cutoff.isoformat())
    new_df = load_training_data(_FEATURE_DIR, start=cutoff)
    new_df = new_df.drop_nulls(subset=FEATURE_COLS + ["label"]).sort("time")

    if len(new_df) == 0:
        logger.info("%s — no new rows since last training, nothing to do", model_name)
        return

    split = int(len(new_df) * _TRAIN_RATIO)
    if split == 0 or split == len(new_df):
        # Too little new data to hold out a test slice — train on all of it,
        # evaluate on the same slice (best effort, small sample anyway).
        train_df = new_df
        test_df = new_df
    else:
        train_df = new_df[:split]
        test_df = new_df[split:]

    X_train = train_df.select(FEATURE_COLS).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test = test_df.select(FEATURE_COLS).to_numpy()
    y_test = test_df["label"].to_numpy()

    model = load_model(model_name, REGISTRY_DIR)
    logger.info("%s — incremental fit on %d new rows (holding out %d for eval)", model_name, len(train_df), len(test_df))
    model.fit_incremental(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluation = evaluate(y_test, y_pred)

    path = save_model(
        model=model,
        evaluation=evaluation,
        params=model.default_params,
        feature_cols=FEATURE_COLS,
        registry_dir=REGISTRY_DIR,
    )
    logger.info(
        "%s — saved incremental update — acc=%.3f prec_buy=%.3f f1_macro=%.3f -> %s",
        model_name, evaluation.accuracy, evaluation.precision_buy, evaluation.f1_macro, path,
    )


def main() -> None:
    for model_name in _MODEL_NAMES:
        try:
            incremental_train_one(model_name)
        except Exception:
            logger.exception("FAILED incremental update for %s", model_name)

    print("\nIncremental training complete. Registry contents:")
    for p in sorted(REGISTRY_DIR.rglob("*.json")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
