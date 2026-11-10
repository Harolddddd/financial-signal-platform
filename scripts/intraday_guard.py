"""Intraday stop checks for the paper accounts (fsp-intraday.timer, every 5 min).

While a market is open, fetch the latest price of each held stock and sell at once
(at that price, with costs) any whose stop-loss or trailing stop is hit, instead of
waiting for the next open. Outside trading hours, or with nothing held, it exits
without touching the network.

Usage: python scripts/intraday_guard.py [--market us|china|all]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.markets import get_market  # noqa: E402
from src.paper.account import PaperConfig, intraday_stop  # noqa: E402

logger = logging.getLogger(__name__)
STATE_FILE = "paper_account.json"
LOCK_FILE = "paper_account.lock"
LOT_SIZE = {"us": None, "china": 100}
MAX_QUOTE_AGE = timedelta(minutes=20)  # older = halted, holiday or a stale feed: do not act on it
_SESSIONS = {
    "us": (ZoneInfo("America/New_York"), [(time(9, 30), time(16, 0))]),
    "china": (ZoneInfo("Asia/Shanghai"), [(time(9, 30), time(11, 30)), (time(13, 0), time(15, 0))]),
}


def in_session(market: str, now: datetime) -> bool:
    tz, spans = _SESSIONS[market]
    local = now.astimezone(tz)
    return local.weekday() < 5 and any(a <= local.time() <= b for a, b in spans)


@contextlib.contextmanager
def account_lock(cache_dir: Path):
    """The daily paper_trade.py run takes the same lock, so the two never write over each other."""
    try:
        import fcntl
    except ImportError:  # Windows dev machine: no other writer runs there
        yield
        return
    with open(cache_dir / LOCK_FILE, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def yahoo_quote(ticker: str) -> tuple[float, datetime] | None:
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="1d", interval="1m")
    if h.empty:
        return None
    ts = h.index[-1].to_pydatetime()
    return float(h["Close"].iloc[-1]), ts.astimezone(timezone.utc)


def check_market(market: str, cache_dir: Path, now: datetime | None = None, quote=yahoo_quote) -> dict:
    now = now or datetime.now(timezone.utc)
    if not in_session(market, now):
        return {"market": market, "skipped": "closed"}
    path = cache_dir / STATE_FILE
    if not path.exists():
        return {"market": market, "skipped": "no account"}
    if not json.loads(path.read_text(encoding="utf-8")).get("positions"):
        return {"market": market, "skipped": "no positions"}
    mcfg = get_market(market)
    cfg = PaperConfig(cost_bps=mcfg.cost_bps, lot_size=LOT_SIZE[market])
    with account_lock(cache_dir):
        acct = json.loads(path.read_text(encoding="utf-8"))  # re-read under the lock
        prices, stale = {}, []
        for t in acct["positions"]:
            try:
                q = quote(t)
            except Exception as exc:  # one bad quote must not stop the others
                logger.warning("intraday %s: quote for %s failed: %s", market, t, exc)
                q = None
            if q is None or now - q[1] > MAX_QUOTE_AGE:
                stale.append(t)
                continue
            prices[t] = q[0]
        day = now.astimezone(_SESSIONS[market][0]).date().isoformat()
        sold = intraday_stop(acct, day, prices, cfg)
        if sold:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(acct, indent=2), encoding="utf-8")
            tmp.replace(path)
    for s in sold:
        logger.warning("intraday %s: SOLD %s %s @ %.2f (%s)", market, s["shares"], s["ticker"], s["price"], s["reason"])
    logger.info("intraday %s: checked %d, sold %d, stale %d", market, len(prices), len(sold), len(stale))
    return {"market": market, "checked": sorted(prices), "sold": [s["ticker"] for s in sold], "stale": stale}


def main(markets: list[str]) -> None:
    for m in markets:
        try:
            check_market(m, get_market(m).data_root / "cache")
        except Exception as exc:
            logger.error("intraday %s failed: %s", m, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="all", choices=["us", "china", "all"])
    a = ap.parse_args()
    main(["us", "china"] if a.market == "all" else [a.market])
