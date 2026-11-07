"""FRED daily macro series + FOMC statement tone -> market-wide US macro features."""

from __future__ import annotations

import html
import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import requests

logger = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")

FRED_SERIES = ("DFF", "DGS2", "DGS10", "T10Y2Y", "BAMLH0A0HYM2")
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

FOMC_MODEL_ID = "gtfintechlab/FOMC-RoBERTa"
# Label mapping from the "Trillion Dollar Words" release; correct here if the model card differs.
FOMC_LABELS = {"LABEL_0": "dovish", "LABEL_1": "hawkish", "LABEL_2": "neutral"}

_FED_BASE = "https://www.federalreserve.gov"
_CALENDAR_URL = f"{_FED_BASE}/monetarypolicy/fomccalendars.htm"
_HISTORICAL_URL = f"{_FED_BASE}/monetarypolicy/fomchistorical{{year}}.htm"
_STATEMENT_RE = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")
_MIN_SENTENCE_CHARS = 20
_NO_FOMC_DAYS = 999

_fomc_pipeline = None


def fetch_fred(series_id: str, start: date, api_key: str, get=requests.get) -> pl.DataFrame:
    resp = get(
        _FRED_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
        },
        timeout=30,
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    return pl.DataFrame(
        {
            "date": [o["date"] for o in obs],
            "value": [None if o["value"] == "." else float(o["value"]) for o in obs],
        },
        schema={"date": pl.String, "value": pl.Float64},
    ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


def _get_fomc_pipeline():
    global _fomc_pipeline
    if _fomc_pipeline is None:
        from transformers import pipeline as hf_pipeline

        _fomc_pipeline = hf_pipeline(
            "text-classification", model=FOMC_MODEL_ID, truncation=True, device=-1
        )
    return _fomc_pipeline


def _classify_sentences(sentences: list[str]) -> list[str]:
    results = _get_fomc_pipeline()(sentences, batch_size=16)
    return [FOMC_LABELS.get(r["label"], "neutral") for r in results]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_SENTENCE_CHARS]


def score_fomc_statements(
    statements: list[tuple[datetime, str]], classify=_classify_sentences
) -> pl.DataFrame:
    """tone = (n_hawkish - n_dovish) / n_sentences per statement. classify: sentences -> labels."""
    stamps, tones = [], []
    for ts, text in statements:
        sentences = _split_sentences(text)
        if not sentences:
            continue
        labels = classify(sentences)
        tones.append((labels.count("hawkish") - labels.count("dovish")) / len(sentences))
        stamps.append(ts)
    return pl.DataFrame(
        {"release_ts": stamps, "tone": tones},
        schema={"release_ts": pl.Datetime("us", "America/New_York"), "tone": pl.Float64},
    )


def _statement_text(page_html: str) -> str:
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", page_html, flags=re.S | re.I)
    cleaned = (html.unescape(re.sub(r"<[^>]+>", "", p)).strip() for p in paragraphs)
    return " ".join(c for c in cleaned if c)


def fetch_fomc_statements(since: date, get=requests.get) -> list[tuple[datetime, str]]:
    """Statements released on/after `since`, 14:00 ET on the meeting's final day."""
    index_urls = [_CALENDAR_URL] + [
        _HISTORICAL_URL.format(year=y) for y in range(since.year, date.today().year + 1)
    ]
    found: dict[str, date] = {}
    for url in index_urls:
        try:
            resp = get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("FOMC index page unavailable %s: %s", url, exc)
            continue
        for stamp in _STATEMENT_RE.findall(resp.text):
            d = datetime.strptime(stamp, "%Y%m%d").date()
            if d >= since:
                found[stamp] = d

    out: list[tuple[datetime, str]] = []
    for stamp in sorted(found):
        d = found[stamp]
        url = f"{_FED_BASE}/newsevents/pressreleases/monetary{stamp}a.htm"
        try:
            resp = get(url, timeout=30)
            resp.raise_for_status()
            text = _statement_text(resp.text)
            if not text:
                raise ValueError("no statement text")
        except Exception as exc:
            logger.warning("FOMC statement skipped %s: %s", url, exc)
            continue
        out.append((datetime(d.year, d.month, d.day, 14, 0, tzinfo=NY), text))
    return out


def _z20(x: pl.Expr) -> pl.Expr:
    z = (x - x.rolling_mean(20)) / x.rolling_std(20)
    return pl.when(z.is_finite()).then(z).otherwise(0.0)


def _fomc_columns(fomc: pl.DataFrame, days: list[date]) -> pl.DataFrame:
    """fomc_tone, fomc_tone_change, days_since_fomc; effective the trading day after release."""
    tone = [0.0] * len(days)
    change = [0.0] * len(days)
    since = [_NO_FOMC_DAYS] * len(days)
    if fomc.height:
        rel_dates = (
            fomc.sort("release_ts")
            .with_columns(pl.col("release_ts").dt.convert_time_zone("America/New_York").dt.date().alias("rd"))
        )
        meetings = list(zip(rel_dates["rd"].to_list(), rel_dates["tone"].to_list()))
        prev_tone = None
        for rd, t in meetings:
            idx = next((i for i, d in enumerate(days) if d > rd), None)
            chg = 0.0 if prev_tone is None else t - prev_tone
            prev_tone = t
            if idx is None:
                continue
            for i in range(idx, len(days)):
                tone[i], change[i], since[i] = t, chg, i - idx
    return pl.DataFrame(
        {"fomc_tone": tone, "fomc_tone_change": change, "days_since_fomc": since},
        schema={"fomc_tone": pl.Float64, "fomc_tone_change": pl.Float64, "days_since_fomc": pl.Int64},
    )


def build_macro_daily(
    fred: dict[str, pl.DataFrame], fomc: pl.DataFrame, trading_days: pl.Series
) -> pl.DataFrame:
    """One row per trading day. FRED columns: forward-filled, derived, then shifted 1 trading day.

    FOMC columns apply their own effective-day rule and are not shifted again.
    """
    base = pl.DataFrame({"date": trading_days.cast(pl.Date)}).unique().sort("date")
    names = {
        "DFF": "macro_dff",
        "DGS2": "macro_dgs2",
        "DGS10": "macro_dgs10",
        "T10Y2Y": "macro_t10y2y",
        "BAMLH0A0HYM2": "macro_hy_spread",
    }
    for sid, col in names.items():
        obs = fred[sid].drop_nulls("value").sort("date").rename({"value": col})
        base = base.join_asof(obs, on="date", strategy="backward")

    base = base.with_columns(
        (pl.col("macro_dgs2") - pl.col("macro_dgs2").shift(5)).alias("macro_dgs2_chg_5d"),
        (pl.col("macro_dgs10") - pl.col("macro_dgs10").shift(5)).alias("macro_dgs10_chg_5d"),
        (pl.col("macro_hy_spread") - pl.col("macro_hy_spread").shift(5)).alias("macro_hy_spread_chg_5d"),
        _z20(pl.col("macro_hy_spread")).alias("macro_hy_spread_z20"),
    )
    fred_cols = [c for c in base.columns if c != "date"]
    base = base.with_columns([pl.col(c).shift(1) for c in fred_cols])
    return base.hstack(_fomc_columns(fomc, base["date"].to_list()))
