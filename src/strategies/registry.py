from __future__ import annotations
import importlib
from pathlib import Path

import yaml

from src.strategies.base import Strategy

_CONFIG_PATH = Path(__file__).parent / "strategies.yaml"


def list_strategies(market: str | None = None) -> list[str]:
    """All strategy names, or only those enabled for `market`.

    An entry may set `markets: [us]` to restrict itself; entries without it run everywhere.
    """
    config = yaml.safe_load(_CONFIG_PATH.read_text())
    return [
        s["name"] for s in config["strategies"]
        if market is None or market in s.get("markets", [market])
    ]


def load_strategy(name: str) -> Strategy:
    config = yaml.safe_load(_CONFIG_PATH.read_text())
    entry = next(s for s in config["strategies"] if s["name"] == name)
    module_path, class_name = entry["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**entry.get("params", {}))
