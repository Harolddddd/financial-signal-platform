from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd


class Signal(Enum):
    BUY  = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


@dataclass
class PredictionResult:
    confidence: pd.Series  # float [0.0, 1.0] per row
    signal: pd.Series      # str: "Buy" / "Hold" / "Sell" per row


@dataclass
class LiveSignal:
    ticker: str
    date: str
    signal: Signal
    confidence: float
    entry_price: float
    position_size: float


class Strategy(ABC):
    data_source: Literal["ohlcv", "features"]

    def fit(self, df: pd.DataFrame) -> None:
        pass  # no-op for rule-based; override in statistical

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        ...
