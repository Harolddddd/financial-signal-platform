from pathlib import Path

from streamlit.testing.v1 import AppTest


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


def test_app_py_renders_market_selector():
    # app.py runs Streamlit calls at module level, so it can't be imported
    # directly in a test — verify the source text instead, same approach
    # already used for dashboard/pages/4_Live_Signals.py in test_data_paths.py.
    source = Path("dashboard/app.py").read_text()
    assert "render_market_selector" in source


def test_data_overview_page_is_market_aware():
    source = Path("dashboard/pages/1_Data_Overview.py").read_text()
    assert "get_selected_market" in source
    assert "get_paths(" in source
    assert "from dashboard.ui_config import PARQUET_DIR" not in source


def test_leaderboard_page_is_market_aware():
    source = Path("dashboard/pages/2_Model_Leaderboard.py").read_text()
    assert "get_selected_market" in source
    assert "get_paths(" in source
    assert "from dashboard.ui_config import PARQUET_DIR" not in source


def test_backtest_results_page_is_market_aware():
    source = Path("dashboard/pages/3_Backtest_Results.py").read_text()
    assert "get_selected_market" in source
    assert "get_paths(" in source
    assert "from dashboard.ui_config import PARQUET_DIR" not in source


def test_live_signals_page_is_market_aware():
    source = Path("dashboard/pages/4_Live_Signals.py").read_text()
    assert "get_selected_market" in source
    assert "get_paths(" in source
    assert 'CACHE_DIR = get_market("us")' not in source
    assert "format_price" in source


def test_combined_signal_page_is_market_aware():
    source = Path("dashboard/pages/5_Combined_Signal.py").read_text()
    assert "get_selected_market" in source
    assert "get_combined_ratings(market=market)" in source
    assert "format_price" in source


def test_market_selection_persists_across_page_navigation():
    # Regression test for a Critical whole-branch-review finding: the market
    # selector widget on app.py is rendered with key="market", and Streamlit
    # garbage-collects a widget-keyed session_state entry the instant that
    # widget stops being rendered (i.e. the moment the user navigates off
    # app.py to any other page). A get_selected_market() that merely *reads*
    # session_state without writing it back will silently see the key gone
    # on the very next page load and fall back to the "us" default — even
    # though the dropdown was switched to "china". Per-page AppTest checks in
    # isolation (see the *_is_market_aware tests above) cannot catch this,
    # because they pre-seed session_state directly rather than navigating
    # through app.py's widget — this test exercises one continuous session
    # that actually switches the dropdown and then navigates, which is the
    # only way to reproduce the bug.
    at = AppTest.from_file("dashboard/app.py", default_timeout=60)
    at.run()
    assert not at.exception

    at.selectbox(key="market").select("china").run()
    assert not at.exception
    assert any("China" in c.value for c in at.caption)

    pages = [
        "pages/1_Data_Overview.py",
        "pages/2_Model_Leaderboard.py",
        "pages/3_Backtest_Results.py",
        "pages/4_Live_Signals.py",
        "pages/5_Combined_Signal.py",
    ]
    for page in pages:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, at.exception)
        captions = [c.value for c in at.caption]
        assert any("China" in c for c in captions), (page, captions)

    # Revisiting the home page itself must also still show China — the
    # dropdown's own session_state must not have been reset either.
    at.switch_page("app.py")
    at.run()
    assert not at.exception
    assert any("China" in c.value for c in at.caption)

    # An in-page rerun (triggered by a widget other than the market
    # selector, e.g. the confidence-threshold slider on Live Signals) must
    # not lose the market selection either.
    at.switch_page("pages/4_Live_Signals.py")
    at.run()
    assert not at.exception
    sliders = at.slider
    assert len(sliders) > 0
    sliders[0].set_value(0.9).run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("China" in c for c in captions), captions
