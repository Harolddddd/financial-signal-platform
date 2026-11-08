from datetime import date, timedelta

import polars as pl

from src.ingestion.cn_macro_ingest import CN_MACRO_COLS, build_cn_macro_daily, monthly_available


def _days(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _s(pairs):
    return pl.DataFrame({"date": [p[0] for p in pairs], "value": [float(p[1]) for p in pairs]},
                        schema={"date": pl.Date, "value": pl.Float64})


def test_monthly_available_is_month_end_plus_lag():
    assert monthly_available("2026年10月份", 15) == date(2026, 11, 15)


def test_monthly_series_not_visible_before_release():
    raw = {"cpi_yoy": _s([(monthly_available("2026年09月份", 15), 0.9),
                          (monthly_available("2026年10月份", 15), 1.2)])}
    out = build_cn_macro_daily(raw, _days(date(2026, 11, 10), 10))
    row = lambda d: out.filter(pl.col("date") == d)["cn_cpi_yoy"].item()
    assert row(date(2026, 11, 13)) == 0.9   # Friday before the Nov 15 release
    assert row(date(2026, 11, 16)) == 1.2


def test_rrr_cut_on_weekend_flags_next_trading_day_for_20_rows():
    days = _days(date(2026, 1, 5), 40)
    raw = {"rrr": _s([(date(2025, 6, 1), 9.5), (date(2026, 1, 10), 9.0)])}  # Sat cut
    out = build_cn_macro_daily(raw, days)
    flags = out["cn_rrr_cut_20d"].to_list()
    first = days.index(date(2026, 1, 12))
    assert flags[first - 1] == 0.0 and flags[first] == 1.0
    assert flags[first + 19] == 1.0 and flags[first + 20] == 0.0


def test_missing_series_gives_all_columns_null():
    out = build_cn_macro_daily({}, _days(date(2026, 1, 5), 5))
    assert out.columns == ["date", *CN_MACRO_COLS]
    assert out["cn_pmi"].null_count() == 5
