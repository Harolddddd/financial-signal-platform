from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import joblib

from src.models.base_classifier import BaseClassifier, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ModelRecord:
    model_name: str
    version: str
    params: dict[str, Any]
    evaluation: EvaluationResult
    feature_cols: list[str]
    trained_at: str


def save_model(
    model: BaseClassifier,
    evaluation: EvaluationResult,
    params: dict[str, Any],
    feature_cols: list[str],
    registry_dir: Path,
) -> Path:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    model_dir = registry_dir / model.name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = model_dir / f"{version}.joblib"
    json_path = model_dir / f"{version}.json"

    joblib.dump(model, joblib_path)
    meta = {
        "model_name": model.name,
        "version": version,
        "params": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                   for k, v in params.items()},
        "evaluation": {
            "accuracy": evaluation.accuracy,
            "precision_buy": evaluation.precision_buy,
            "recall_buy": evaluation.recall_buy,
            "f1_buy": evaluation.f1_buy,
            "f1_macro": evaluation.f1_macro,
            "confusion_matrix": evaluation.confusion_matrix,
            "class_labels": evaluation.class_labels,
            "n_samples": evaluation.n_samples,
        },
        "feature_cols": feature_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s v%s → %s", model.name, version, joblib_path)
    return joblib_path


def load_model(
    model_name: str,
    registry_dir: Path,
    version: str | None = None,
) -> BaseClassifier:
    model_dir = registry_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No saved versions for '{model_name}' in {registry_dir}")

    versions = sorted(model_dir.glob("*.joblib"))
    if not versions:
        raise FileNotFoundError(f"No saved versions for '{model_name}' in {registry_dir}")

    if version:
        path = model_dir / f"{version}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Version {version} not found for '{model_name}'")
    else:
        path = versions[-1]

    return joblib.load(path)


def list_models(registry_dir: Path) -> list[ModelRecord]:
    records: list[ModelRecord] = []
    for json_path in sorted(registry_dir.rglob("*.json"), reverse=True):
        try:
            meta = json.loads(json_path.read_text())
            ev = meta["evaluation"]
            records.append(ModelRecord(
                model_name=meta["model_name"],
                version=meta["version"],
                params=meta["params"],
                evaluation=EvaluationResult(
                    accuracy=ev["accuracy"],
                    precision_buy=ev["precision_buy"],
                    recall_buy=ev["recall_buy"],
                    f1_buy=ev["f1_buy"],
                    f1_macro=ev["f1_macro"],
                    confusion_matrix=ev["confusion_matrix"],
                    class_labels=ev["class_labels"],
                    n_samples=ev["n_samples"],
                ),
                feature_cols=meta["feature_cols"],
                trained_at=meta["trained_at"],
            ))
        except Exception as e:
            logger.warning("Skipping malformed record %s: %s", json_path, e)
    return records
