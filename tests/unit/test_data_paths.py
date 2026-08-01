from pathlib import Path

from config.markets import get_market

_US_ROOT = get_market("us").data_root


def test_dashboard_config_paths_resolve_under_markets_us_data():
    from dashboard.config import PARQUET_DIR, REGISTRY_DIR
    assert PARQUET_DIR == _US_ROOT / "features"
    assert REGISTRY_DIR == _US_ROOT / "registry"
    assert PARQUET_DIR.exists()
    assert REGISTRY_DIR.exists()


def test_dashboard_data_loader_cache_dir_resolves_under_markets_us_data():
    from dashboard.data_loader import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"
    assert CACHE_DIR.exists()


def test_live_signals_page_no_longer_hardcodes_old_data_path():
    # dashboard/pages/4_Live_Signals.py runs Streamlit UI calls
    # (st.set_page_config, st.slider, ...) at module level, and "4_Live_Signals"
    # isn't a valid Python identifier — it can't be imported directly in a
    # test. Verify the source text instead: the stale literal must be gone
    # and the fix must route through get_market.
    source = Path("dashboard/pages/4_Live_Signals.py").read_text()
    assert 'Path("data/cache")' not in source
    assert "get_market(" in source
