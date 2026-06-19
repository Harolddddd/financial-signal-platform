import pandas as pd
import pytest
from src.strategies.base import Signal, PredictionResult, Strategy, LiveSignal


def test_signal_enum_values():
    assert Signal.BUY.value == "Buy"
    assert Signal.HOLD.value == "Hold"
    assert Signal.SELL.value == "Sell"


def test_prediction_result_holds_series():
    conf = pd.Series([0.8, 0.2, 0.5])
    sig = pd.Series(["Buy", "Hold", "Hold"])
    result = PredictionResult(confidence=conf, signal=sig)
    assert len(result.confidence) == 3
    assert len(result.signal) == 3


def test_strategy_is_abstract():
    with pytest.raises(TypeError):
        Strategy()


def test_strategy_fit_is_noop_by_default():
    class ConcreteStrategy(Strategy):
        data_source = "ohlcv"
        def predict(self, df: pd.DataFrame):
            return PredictionResult(
                confidence=pd.Series([0.5] * len(df)),
                signal=pd.Series(["Hold"] * len(df)),
            )

    s = ConcreteStrategy()
    df = pd.DataFrame({"close": [100.0, 101.0]})
    s.fit(df)   # must not raise


def test_live_signal_fields():
    sig = LiveSignal(
        ticker="AAPL",
        date="2024-01-01",
        signal=Signal.BUY,
        confidence=0.82,
        entry_price=195.0,
        position_size=0.82,
    )
    assert sig.ticker == "AAPL"
    assert sig.signal == Signal.BUY
    assert sig.confidence == 0.82
