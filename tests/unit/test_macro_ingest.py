from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.ingestion import macro_ingest
from src.ingestion.macro_ingest import (
    build_macro_daily,
    fetch_fomc_statements,
    fetch_fred,
    score_fomc_statements,
)

NY = ZoneInfo("America/New_York")

MACRO_COLS = [
    "macro_dff", "macro_dgs2", "macro_dgs10", "macro_t10y2y", "macro_hy_spread",
    "macro_dgs2_chg_5d", "macro_dgs10_chg_5d", "macro_hy_spread_chg_5d",
    "macro_hy_spread_z20", "fomc_tone", "fomc_tone_change", "days_since_fomc",
]


class _Resp:
    def __init__(self, payload=None, text="", status=200):
        self._payload, self.text, self.status_code = payload, text, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _series(pairs):
    return pl.DataFrame({"date": [p[0] for p in pairs], "value": [p[1] for p in pairs]}).with_columns(
        pl.col("date").cast(pl.Date), pl.col("value").cast(pl.Float64)
    )


def _empty_fomc():
    return pl.DataFrame(schema={"release_ts": pl.Datetime("us", "America/New_York"), "tone": pl.Float64})


def test_fetch_fred_parses_dots_as_null():
    seen = {}

    def get(url, params=None, timeout=None):
        seen["params"] = params
        return _Resp({"observations": [
            {"date": "2024-01-02", "value": "4.33"},
            {"date": "2024-01-03", "value": "."},
        ]})

    df = fetch_fred("DGS10", date(2024, 1, 1), "KEY", get=get)
    assert df.columns == ["date", "value"]
    assert df["value"].dtype == pl.Float64
    assert df["value"].to_list() == [4.33, None]
    assert df["date"].dtype == pl.Date
    assert seen["params"]["series_id"] == "DGS10"
    assert seen["params"]["observation_start"] == "2024-01-01"


def test_macro_forward_fill_and_shift():
    days = pl.Series([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)])
    s = _series([(date(2024, 1, 2), 1.0), (date(2024, 1, 4), 3.0)])
    fred = {k: s for k in macro_ingest.FRED_SERIES}
    out = build_macro_daily(fred, _empty_fomc(), days)
    assert out.columns == ["date"] + MACRO_COLS
    assert out["date"].to_list() == days.to_list()
    # day T value appears on T+1; gap day (01-03) is forward-filled to 1.0
    assert out["macro_dgs10"].to_list() == [None, 1.0, 1.0, 3.0]


def test_fomc_effective_next_trading_day():
    days = pl.Series([date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 2)])
    s = _series([(date(2024, 1, 30), 1.0)])
    fred = {k: s for k in macro_ingest.FRED_SERIES}
    fomc = pl.DataFrame({"release_ts": [datetime(2024, 1, 31, 14, 0, tzinfo=NY)], "tone": [0.4]})
    out = build_macro_daily(fred, fomc, days)
    assert out["days_since_fomc"].to_list() == [999, 999, 0, 1]
    assert out["fomc_tone"].to_list() == [0.0, 0.0, 0.4, 0.4]
    assert out["fomc_tone_change"].to_list() == [0.0, 0.0, 0.0, 0.0]


def test_fomc_tone_change_vs_previous_meeting():
    days = pl.Series([date(2024, 1, 31), date(2024, 2, 1), date(2024, 3, 20), date(2024, 3, 21)])
    s = _series([(date(2024, 1, 30), 1.0)])
    fred = {k: s for k in macro_ingest.FRED_SERIES}
    fomc = pl.DataFrame({
        "release_ts": [datetime(2024, 1, 31, 14, tzinfo=NY), datetime(2024, 3, 20, 14, tzinfo=NY)],
        "tone": [0.5, 0.25],
    })
    out = build_macro_daily(fred, fomc, days)
    assert out["fomc_tone_change"].to_list() == [0.0, 0.0, 0.0, -0.25]
    assert out["fomc_tone"].to_list() == [0.0, 0.5, 0.5, 0.25]


def test_fomc_tone_formula():
    labels = ["hawkish", "hawkish", "dovish", "neutral"]
    text = " ".join(f"This is sentence number {i} of the statement." for i in range(4))
    calls = []

    def classify(sentences):
        calls.append(sentences)
        return labels

    ts = datetime(2024, 1, 31, 14, tzinfo=NY)
    out = score_fomc_statements([(ts, text)], classify=classify)
    assert out.columns == ["release_ts", "tone"]
    assert out["tone"].to_list() == [0.25]
    assert len(calls[0]) == 4


def test_label_mapping_constant():
    assert macro_ingest.FOMC_LABELS == {"LABEL_0": "dovish", "LABEL_1": "hawkish", "LABEL_2": "neutral"}


def test_fetch_fomc_statements_canned_html():
    index = (
        '<a href="/newsevents/pressreleases/monetary20240131a.htm">s</a>'
        '<a href="/newsevents/pressreleases/monetary20230201a.htm">old</a>'
        '<a href="/newsevents/pressreleases/monetary20240320a.htm">bad</a>'
    )
    page = "<html><body><p>Inflation remains elevated across the economy.</p><p>Short.</p></body></html>"

    def get(url, **kw):
        if url.endswith("fomccalendars.htm"):
            return _Resp(text=index)
        if "20240131" in url:
            return _Resp(text=page)
        if "20240320" in url:
            return _Resp(text="", status=500)
        return _Resp(text="", status=404)

    out = fetch_fomc_statements(date(2024, 1, 1), get=get)
    assert len(out) == 1
    ts, text = out[0]
    assert ts == datetime(2024, 1, 31, 14, 0, tzinfo=NY)
    assert "Inflation remains elevated" in text


def test_derived_columns_values_and_shift():
    n = 30
    days = pl.Series([date(2024, 1, 1) + timedelta(days=i) for i in range(n)])
    vals = [float(i * i) for i in range(n)]  # x_i = i^2
    s = _series([(d, v) for d, v in zip(days.to_list(), vals)])
    fred = {k: s for k in macro_ingest.FRED_SERIES}
    out = build_macro_daily(fred, _empty_fomc(), days)

    # row r holds the derived value of row r-1 (1-trading-day shift)
    r = 27
    assert out["macro_dgs10_chg_5d"][r] == vals[r - 1] - vals[r - 6]
    assert out["macro_hy_spread_chg_5d"][r] == vals[r - 1] - vals[r - 6]
    window = vals[r - 20:r]  # x_{r-20}..x_{r-1}
    mean = sum(window) / 20
    std = (sum((v - mean) ** 2 for v in window) / 19) ** 0.5
    assert abs(out["macro_hy_spread_z20"][r] - (vals[r - 1] - mean) / std) < 1e-9
    # chg_5d undefined for first 5 rows (+1 shift): rows 0..5 null; z20 undefined -> 0.0, shifted
    assert out["macro_dgs10_chg_5d"].head(6).to_list() == [None] * 6
    assert out["macro_hy_spread_z20"][0] is None
    assert out["macro_hy_spread_z20"][1] == 0.0
    assert out["macro_hy_spread_z20"][19] == 0.0
    assert out["macro_hy_spread_z20"][20] != 0.0


def test_fomc_scoring_skips_short_sentences_and_empty_statements():
    seen = []

    def classify(sentences):
        seen.append(list(sentences))
        return ["hawkish"] * len(sentences)

    ts1 = datetime(2024, 1, 31, 14, tzinfo=NY)
    ts2 = datetime(2024, 3, 20, 14, tzinfo=NY)
    text = "Too short. This sentence is long enough to count here."
    out = score_fomc_statements([(ts1, text), (ts2, "")], classify=classify)
    assert seen == [["This sentence is long enough to count here."]]
    assert out["tone"].to_list() == [1.0]
    assert out.height == 1


def test_fomc_calendar_page_failure_warns(caplog):
    def get(url, **kw):
        return _Resp(text="", status=500)

    with caplog.at_level("WARNING"):
        assert fetch_fomc_statements(date(2024, 1, 1), get=get) == []
    assert any("fomccalendars" in r.message for r in caplog.records if r.levelname == "WARNING")


def test_statement_text_keeps_only_article_body():
    page = (
        "<p>An official website of the United States Government</p>"
        '<div id="article"><p>The Committee decided to maintain the target range.</p></div>'
        '<div class="footer__footer"><p>Last Update: today</p></div>'
    )
    assert macro_ingest._statement_text(page) == "The Committee decided to maintain the target range."


def test_statement_text_falls_back_to_whole_page_without_article_div():
    assert macro_ingest._statement_text("<p>First.</p><p>Second.</p>") == "First. Second."
