import yaml

from scripts.build_features import _STOCK_TICKERS_CHINA

MAIN_BOARD = ("600", "601", "603", "605", "000", "001", "002", "003")


def test_china_universe_is_main_board_only():
    # The account cannot trade ChiNext (300/301) or STAR (688/689).
    assert all(t[:3] in MAIN_BOARD for t in _STOCK_TICKERS_CHINA)
    assert len(set(_STOCK_TICKERS_CHINA)) == len(_STOCK_TICKERS_CHINA) == 500


def test_stocks_yaml_matches_code_list():
    y = yaml.safe_load(open("config/stocks.yaml", encoding="utf-8"))["universes"]["csi500"]["tickers"]
    assert sorted(y) == sorted(_STOCK_TICKERS_CHINA)
