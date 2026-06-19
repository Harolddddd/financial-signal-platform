# tests/unit/test_rule_based_strategies.py
import numpy as np
import pandas as pd
import pytest

from src.strategies.rule_based.ma_crossover import MACrossover
from src.strategies.rule_based.rsi import RSIThreshold
from src.strategies.rule_based.macd import MACDSignal
from src.strategies.rule_based.bollinger import BollingerBounce
from src.strategies.base import PredictionResult

_VALID_SIGNALS = {"Buy", "Hold", "Sell"}


def _uptrend_df(n: int = 200) -> pd.DataFrame:
    """Strong uptrend: price rises 0.5/day."""
    closes = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame({
        "close": closes,
        "open":  [c - 0.2 for c in closes],
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.5 for c in closes],
        "volume": [1_000_000] * n,
    })


def _downtrend_df(n: int = 100) -> pd.DataFrame:
    """Sharp downtrend: price falls 1.0/day (creates oversold RSI)."""
    closes = [200.0 - i * 1.0 for i in range(n)]
    return pd.DataFrame({
        "close": closes,
        "open":  [c + 0.2 for c in closes],
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.5 for c in closes],
        "volume": [1_000_000] * n,
    })


def _flat_then_surge_df(flat: int = 60, surge: int = 20) -> pd.DataFrame:
    """Flat for `flat` days then sharp up for `surge` days — triggers MACD."""
    flat_closes = [100.0] * flat
    surge_closes = [100.0 + i * 2.0 for i in range(1, surge + 1)]
    closes = flat_closes + surge_closes
    return pd.DataFrame({
        "close": closes,
        "open":  [c - 0.2 for c in closes],
        "high":  [c + 0.5 for c in closes],
        "low":   [c - 0.5 for c in closes],
        "volume": [1_000_000] * len(closes),
    })


# --- MACrossover ---

def test_ma_crossover_returns_prediction_result():
    df = _uptrend_df()
    result = MACrossover(fast_window=20, slow_window=50).predict(df)
    assert isinstance(result, PredictionResult)
    assert len(result.confidence) == len(df)
    assert len(result.signal) == len(df)


def test_ma_crossover_confidence_in_range():
    result = MACrossover().predict(_uptrend_df())
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_ma_crossover_signals_are_valid():
    result = MACrossover().predict(_uptrend_df())
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_ma_crossover_buys_in_uptrend():
    result = MACrossover(fast_window=20, slow_window=50).predict(_uptrend_df(200))
    # After warmup (50 days), fast > slow → BUY
    assert result.signal.iloc[-1] == "Buy"


# --- RSIThreshold ---

def test_rsi_returns_prediction_result():
    result = RSIThreshold().predict(_downtrend_df())
    assert isinstance(result, PredictionResult)
    assert len(result.confidence) == len(_downtrend_df())


def test_rsi_confidence_in_range():
    result = RSIThreshold().predict(_downtrend_df())
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_rsi_signals_are_valid():
    result = RSIThreshold().predict(_downtrend_df())
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_rsi_buys_when_oversold():
    result = RSIThreshold(period=14, oversold=30).predict(_downtrend_df(60))
    # Sharp downtrend pushes RSI below 30 → BUY signal appears
    assert "Buy" in result.signal.values


# --- MACDSignal ---

def test_macd_returns_prediction_result():
    result = MACDSignal().predict(_flat_then_surge_df())
    assert isinstance(result, PredictionResult)


def test_macd_confidence_in_range():
    result = MACDSignal().predict(_flat_then_surge_df())
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_macd_signals_are_valid():
    result = MACDSignal().predict(_flat_then_surge_df())
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_macd_buys_after_surge():
    result = MACDSignal(fast=12, slow=26, signal_period=9).predict(_flat_then_surge_df())
    # Momentum surge causes MACD to cross above signal line
    assert "Buy" in result.signal.values


# --- BollingerBounce ---

def test_bollinger_returns_prediction_result():
    result = BollingerBounce().predict(_downtrend_df())
    assert isinstance(result, PredictionResult)


def test_bollinger_confidence_in_range():
    result = BollingerBounce().predict(_downtrend_df())
    assert (result.confidence >= 0.0).all()
    assert (result.confidence <= 1.0).all()


def test_bollinger_signals_are_valid():
    result = BollingerBounce().predict(_downtrend_df())
    assert set(result.signal.unique()).issubset(_VALID_SIGNALS)


def test_bollinger_buys_when_below_lower_band():
    result = BollingerBounce(window=20, num_std=2.0).predict(_downtrend_df(60))
    # Price drops sharply below lower band
    assert "Buy" in result.signal.values
