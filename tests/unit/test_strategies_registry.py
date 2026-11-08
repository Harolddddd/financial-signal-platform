import pytest
from src.strategies.registry import list_strategies, load_strategy
from src.strategies.base import Strategy


def test_list_strategies_returns_list():
    names = list_strategies()
    assert isinstance(names, list)
    assert len(names) >= 1


def test_list_strategies_contains_expected():
    names = list_strategies()
    assert "momentum" in names
    assert "meta_labeling" in names
    assert "mean_reversion" in names
    assert "bollinger_bounce" in names
    assert "logistic_regression" in names
    assert "linear_regression" in names


def test_load_strategy_returns_strategy_instance():
    strategy = load_strategy("momentum")
    assert isinstance(strategy, Strategy)


def test_load_strategy_injects_params():
    strategy = load_strategy("momentum")
    assert strategy.period == 20
    assert strategy.threshold == 0.03


def test_load_strategy_unknown_raises():
    with pytest.raises((KeyError, StopIteration)):
        load_strategy("does_not_exist")
