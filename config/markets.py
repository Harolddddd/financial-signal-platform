from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketConfig:
    name: str
    label: str
    data_root: Path
    universe: str
    benchmark_ticker: str
    vol_index_ticker: str | None
    currency: str


MARKETS: dict[str, MarketConfig] = {
    "us": MarketConfig(
        name="us",
        label="United States (S&P 500)",
        data_root=Path("markets/us/data"),
        universe="sp500",
        benchmark_ticker="SPY",
        vol_index_ticker="^VIX",
        currency="USD",
    ),
    "china": MarketConfig(
        name="china",
        label="China A-Share (CSI 300)",
        data_root=Path("markets/china/data"),
        universe="csi300",
        benchmark_ticker="000300.SS",
        vol_index_ticker=None,
        currency="CNY",
    ),
}


def get_market(name: str) -> MarketConfig:
    try:
        return MARKETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown market {name!r}; valid markets: {sorted(MARKETS)}"
        ) from None
