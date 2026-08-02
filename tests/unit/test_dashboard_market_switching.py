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


def test_data_loader_get_cache_dir_for_us():
    from dashboard.data_loader import get_cache_dir
    from config.markets import get_market
    assert get_cache_dir("us") == get_market("us").data_root / "cache"


def test_data_loader_get_cache_dir_for_china():
    from dashboard.data_loader import get_cache_dir
    from config.markets import get_market
    assert get_cache_dir("china") == get_market("china").data_root / "cache"


def test_get_data_summary_routes_to_china_cache():
    from dashboard.data_loader import get_data_summary
    from dashboard.ui_config import get_paths
    parquet_dir, _ = get_paths("china")
    summary = get_data_summary(parquet_dir, market="china")
    assert summary["n_tickers"] == 500


def test_get_data_summary_routes_to_us_cache_by_default():
    from dashboard.data_loader import get_data_summary
    from dashboard.ui_config import PARQUET_DIR
    summary = get_data_summary(PARQUET_DIR)
    assert summary["n_tickers"] == 492


def test_get_leaderboard_routes_to_china_cache():
    from dashboard.data_loader import get_leaderboard
    from dashboard.ui_config import get_paths, OHLCV_COLS, FEATURE_COLS
    parquet_dir, _ = get_paths("china")
    leaderboard = get_leaderboard(parquet_dir, OHLCV_COLS, FEATURE_COLS, market="china")
    assert len(leaderboard) == 32


def test_get_combined_ratings_routes_to_china_cache():
    from dashboard.data_loader import get_combined_ratings
    summary_rows, detail_by_ticker = get_combined_ratings(market="china")
    assert len(summary_rows) > 0
    assert isinstance(detail_by_ticker, dict)


from pathlib import Path


def test_app_py_renders_market_selector():
    # app.py runs Streamlit calls at module level, so it can't be imported
    # directly in a test — verify the source text instead, same approach
    # already used for dashboard/pages/4_Live_Signals.py in test_data_paths.py.
    source = Path("dashboard/app.py").read_text()
    assert "render_market_selector" in source
