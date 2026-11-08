from __future__ import annotations
from dataclasses import dataclass
from datetime import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MarketConfig:
    name: str
    label: str
    data_root: Path
    universe: str
    benchmark_ticker: str
    vol_index_ticker: str | None
    currency: str
    default_strategies: tuple[str, ...] | None = None
    timezone: str = "America/New_York"
    close_time: time = time(16, 0)
    # Per-side trading cost in basis points (China includes stamp duty).
    cost_bps: float = 10.0


MARKETS: dict[str, MarketConfig] = {
    "us": MarketConfig(
        name="us",
        label="United States (S&P 500)",
        data_root=_REPO_ROOT / "markets" / "us" / "data",
        universe="sp500",
        benchmark_ticker="SPY",
        vol_index_ticker="^VIX",
        currency="USD",
        default_strategies=("extra_trees", "lda_strategy", "random_forest", "conformal_lgbm"),
    ),
    "china": MarketConfig(
        name="china",
        label="China A-Share (CSI 500)",
        data_root=_REPO_ROOT / "markets" / "china" / "data",
        universe="csi500",
        benchmark_ticker="510300.SS",
        vol_index_ticker=None,
        currency="CNY",
        timezone="Asia/Shanghai",
        close_time=time(15, 0),
        cost_bps=15.0,
    ),
}


def get_market(name: str) -> MarketConfig:
    try:
        return MARKETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown market {name!r}; valid markets: {sorted(MARKETS)}"
        ) from None
