import json

from config.markets import get_market
from src.strategies.registry import list_strategies

_CHINA_REGISTRY_DIR = get_market("china").data_root / "registry"
_CHINA_CACHE_DIR = get_market("china").data_root / "cache"
_EXPECTED_MODELS = ["random_forest", "xgboost", "lightgbm"]


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def test_china_registry_has_all_three_core_models():
    for name in _EXPECTED_MODELS:
        model_dir = _CHINA_REGISTRY_DIR / name
        assert model_dir.exists(), f"no registry dir for {name}"
        json_files = list(model_dir.glob("*.json"))
        assert len(json_files) > 0, f"no saved model json for {name}"


def test_china_cache_has_leaderboard_and_signals():
    leaderboard_path = _CHINA_CACHE_DIR / "leaderboard.json"
    signals_path = _CHINA_CACHE_DIR / "signals.json"
    assert leaderboard_path.exists()
    assert signals_path.exists()

    leaderboard = json.loads(leaderboard_path.read_text())
    assert len(leaderboard["grades"]) > 0

    signals = json.loads(signals_path.read_text())
    assert len(signals["signals"]) > 0


def test_china_cache_has_backtest_file_per_strategy():
    for name in list_strategies():
        backtest_path = _CHINA_CACHE_DIR / f"backtest_{_safe(name)}.json"
        assert backtest_path.exists(), f"missing backtest cache for {name}"
