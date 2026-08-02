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
