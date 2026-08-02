def test_get_selected_market_defaults_to_us(monkeypatch):
    from dashboard import market_state
    monkeypatch.setattr(market_state.st, "session_state", {})
    assert market_state.get_selected_market() == "us"


def test_get_selected_market_returns_stored_value(monkeypatch):
    from dashboard import market_state
    monkeypatch.setattr(market_state.st, "session_state", {"market": "china"})
    assert market_state.get_selected_market() == "china"


def test_format_price_usd():
    from dashboard.market_state import format_price
    assert format_price(123.456, "us") == "$123.46"


def test_format_price_cny():
    from dashboard.market_state import format_price
    assert format_price(123.456, "china") == "¥123.46"


def test_ui_config_get_paths_for_us():
    from dashboard.ui_config import get_paths
    from config.markets import get_market
    feature_dir, registry_dir = get_paths("us")
    assert feature_dir == get_market("us").data_root / "features"
    assert registry_dir == get_market("us").data_root / "registry"


def test_ui_config_get_paths_for_china():
    from dashboard.ui_config import get_paths
    from config.markets import get_market
    feature_dir, registry_dir = get_paths("china")
    assert feature_dir == get_market("china").data_root / "features"
    assert registry_dir == get_market("china").data_root / "registry"


def test_ui_config_paths_still_exported_for_other_scripts():
    # scripts/train_models.py and 7 others import PARQUET_DIR/REGISTRY_DIR
    # directly — this must keep working unchanged.
    from dashboard.ui_config import PARQUET_DIR, REGISTRY_DIR
    from config.markets import get_market
    assert PARQUET_DIR == get_market("us").data_root / "features"
    assert REGISTRY_DIR == get_market("us").data_root / "registry"
