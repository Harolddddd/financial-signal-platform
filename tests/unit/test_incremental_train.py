def test_market_paths_for_us():
    from scripts.incremental_train import _market_paths
    from config.markets import get_market
    feature_dir, registry_dir = _market_paths("us")
    assert feature_dir == get_market("us").data_root / "features"
    assert registry_dir == get_market("us").data_root / "registry"


def test_market_paths_for_china():
    from scripts.incremental_train import _market_paths
    from config.markets import get_market
    feature_dir, registry_dir = _market_paths("china")
    assert feature_dir == get_market("china").data_root / "features"
    assert registry_dir == get_market("china").data_root / "registry"
