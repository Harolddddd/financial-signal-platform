from pathlib import Path

from config.markets import get_market

_US_ROOT = get_market("us").data_root


def test_dashboard_ui_config_paths_resolve_under_markets_us_data():
    from dashboard.ui_config import PARQUET_DIR, REGISTRY_DIR
    assert PARQUET_DIR == _US_ROOT / "features"
    assert REGISTRY_DIR == _US_ROOT / "registry"
    assert PARQUET_DIR.exists()
    assert REGISTRY_DIR.exists()


def test_dashboard_data_loader_cache_dir_resolves_under_markets_us_data():
    from dashboard.data_loader import get_cache_dir
    assert get_cache_dir("us") == _US_ROOT / "cache"
    assert get_cache_dir("us").exists()


def test_live_signals_page_no_longer_hardcodes_old_data_path():
    # dashboard/pages/4_Live_Signals.py runs Streamlit UI calls
    # (st.set_page_config, st.slider, ...) at module level, and "4_Live_Signals"
    # isn't a valid Python identifier — it can't be imported directly in a
    # test. Verify the source text instead: the stale literal must be gone
    # and the fix must route through get_market.
    source = Path("dashboard/pages/4_Live_Signals.py").read_text()
    assert 'Path("data/cache")' not in source
    assert "get_market(" in source


def test_scrape_top20_output_dir_resolves_under_markets_us_data():
    from scripts.scrape_top20 import _OUTPUT_DIR
    assert _OUTPUT_DIR == _US_ROOT / "raw" / "ohlcv"


def test_build_features_dirs_resolve_under_markets_us_data():
    from scripts.build_features import _RAW_DIR, _FEATURE_DIR
    assert _RAW_DIR == _US_ROOT / "raw" / "ohlcv"
    assert _FEATURE_DIR == _US_ROOT / "features"
    assert _FEATURE_DIR.exists()


def test_refresh_data_dirs_resolve_under_markets_us_data():
    from scripts.refresh_data import _RAW_DIR, _FEATURE_DIR
    assert _RAW_DIR == _US_ROOT / "raw" / "ohlcv"
    assert _FEATURE_DIR == _US_ROOT / "features"


def test_incremental_train_feature_dir_resolves_under_markets_us_data():
    from scripts.incremental_train import _FEATURE_DIR
    assert _FEATURE_DIR == _US_ROOT / "features"


def test_train_new_models_feature_dir_resolves_under_markets_us_data():
    from scripts.train_new_models import _FEATURE_DIR
    assert _FEATURE_DIR == _US_ROOT / "features"


def test_train_lstm_only_feature_dir_resolves_under_markets_us_data():
    from scripts.train_lstm_only import _FEATURE_DIR
    assert _FEATURE_DIR == _US_ROOT / "features"


def test_train_models_feature_dir_resolves_under_markets_us_data():
    from scripts.train_models import _FEATURE_DIR
    assert _FEATURE_DIR == _US_ROOT / "features"


def test_precompute_new_strategies_cache_dir_resolves_under_markets_us_data():
    from scripts.precompute_new_strategies import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"
    assert CACHE_DIR.exists()


def test_precompute_dashboard_cache_dir_resolves_under_markets_us_data():
    from scripts.precompute_dashboard import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"


def test_precompute_full_cache_dir_resolves_under_markets_us_data():
    from scripts.precompute_full import CACHE_DIR
    assert CACHE_DIR == _US_ROOT / "cache"


def test_signal_one_strategy_paths_resolve_under_markets_us_data():
    from scripts.signal_one_strategy import _OUT_DIR, _LIVE_CACHE, _TRAIN_CACHE
    assert _OUT_DIR == _US_ROOT / "cache" / "signals_partial"
    assert _LIVE_CACHE == _US_ROOT / "cache" / "_tmp_live_features.parquet"
    assert _TRAIN_CACHE == _US_ROOT / "cache" / "_tmp_training_data.parquet"


def test_run_signals_isolated_paths_resolve_under_markets_us_data():
    from scripts.run_signals_isolated import _PARTIAL_DIR, _LIVE_CACHE, _TRAIN_CACHE, _SIGNALS_PATH
    assert _PARTIAL_DIR == _US_ROOT / "cache" / "signals_partial"
    assert _LIVE_CACHE == _US_ROOT / "cache" / "_tmp_live_features.parquet"
    assert _TRAIN_CACHE == _US_ROOT / "cache" / "_tmp_training_data.parquet"
    assert _SIGNALS_PATH == _US_ROOT / "cache" / "signals.json"


def test_signal_one_strategy_and_run_signals_isolated_agree_on_cache_paths():
    # The parent (run_signals_isolated) writes _LIVE_CACHE/_TRAIN_CACHE for
    # the child (signal_one_strategy, spawned as a subprocess) to read —
    # they must always point at the exact same files.
    from scripts.signal_one_strategy import _LIVE_CACHE as child_live, _TRAIN_CACHE as child_train
    from scripts.run_signals_isolated import _LIVE_CACHE as parent_live, _TRAIN_CACHE as parent_train
    assert child_live == parent_live
    assert child_train == parent_train


def test_precompute_dashboard_market_paths_for_us():
    from scripts.precompute_dashboard import _market_paths
    feature_dir, cache_dir = _market_paths("us")
    assert feature_dir == _US_ROOT / "features"
    assert cache_dir == _US_ROOT / "cache"


def test_precompute_dashboard_market_paths_for_china():
    from scripts.precompute_dashboard import _market_paths
    from config.markets import get_market
    feature_dir, cache_dir = _market_paths("china")
    assert feature_dir == get_market("china").data_root / "features"
    assert cache_dir == get_market("china").data_root / "cache"


def test_dashboard_ui_config_survives_streamlit_sys_path_shadowing():
    # Streamlit inserts the main script's own directory at sys.path[0]
    # (ahead of the repo root) before running it — for `streamlit run
    # dashboard/app.py` that's dashboard/. If any dashboard module needs
    # `from config.markets import get_market` while sitting in a file that
    # could shadow the top-level `config` package name once dashboard/
    # precedes the repo root on sys.path, every dashboard page breaks.
    # Reproduce that exact ordering in a subprocess and confirm the import
    # still resolves cleanly.
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    dashboard_dir = repo_root / "dashboard"
    code = (
        "import sys; "
        f"sys.path.insert(0, r'{dashboard_dir}'); "
        f"sys.path.insert(1, r'{repo_root}'); "
        "import dashboard.ui_config"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
