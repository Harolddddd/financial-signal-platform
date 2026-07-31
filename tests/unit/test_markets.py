from pathlib import Path

import pytest

from config.markets import MARKETS, MarketConfig, get_market


def test_us_market_config():
    us = get_market("us")
    assert isinstance(us, MarketConfig)
    assert us.name == "us"
    assert us.data_root == Path(__file__).resolve().parents[2] / "markets" / "us" / "data"
    assert us.universe == "sp500"
    assert us.benchmark_ticker == "SPY"
    assert us.vol_index_ticker == "^VIX"
    assert us.currency == "USD"


def test_china_market_config():
    china = get_market("china")
    assert china.name == "china"
    assert china.data_root == Path(__file__).resolve().parents[2] / "markets" / "china" / "data"
    assert china.universe == "csi300"
    assert china.benchmark_ticker == "000300.SS"
    assert china.vol_index_ticker is None
    assert china.currency == "CNY"


def test_markets_have_distinct_data_roots():
    roots = {m.data_root for m in MARKETS.values()}
    assert len(roots) == len(MARKETS)


def test_get_market_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown market 'atlantis'"):
        get_market("atlantis")
