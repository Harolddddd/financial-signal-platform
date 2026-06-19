import pytest
from src.strategies.registry import list_strategies, load_strategy
from src.strategies.base import Strategy


def test_list_strategies_returns_list():
    names = list_strategies()
    assert isinstance(names, list)
    assert len(names) >= 1


def test_list_strategies_contains_expected():
    names = list_strategies()
    assert "ma_crossover" in names
    assert "rsi_threshold" in names
    assert "macd_signal" in names
    assert "bollinger_bounce" in names
    assert "logistic_regression" in names
    assert "linear_regression" in names


def test_load_strategy_returns_strategy_instance():
    strategy = load_strategy("ma_crossover")
    assert isinstance(strategy, Strategy)


def test_load_strategy_injects_params():
    strategy = load_strategy("ma_crossover")
    assert strategy.fast_window == 20
    assert strategy.slow_window == 50


def test_load_strategy_unknown_raises():
    with pytest.raises((KeyError, StopIteration)):
        load_strategy("does_not_exist")
