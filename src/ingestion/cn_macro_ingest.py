"""China macro series (akshare) -> one row per trading day, keyed by when each value was known.

`fetch_cn_macro_raw` is the only network code. Every raw series it returns is
(date, value) where `date` is the first day the value was public, so
`build_cn_macro_daily` can forward-fill with no look-ahead.
"""
from __future__ import annotations

import calendar
import logging
import re
from datetime import date, timedelta

import polars as pl

logger = logging.getLogger(__name__)

CN_MACRO_COLS = [
    "cn_lpr_1y", "cn_lpr_1y_chg_20d", "cn_rrr", "cn_rrr_cut_20d",
    "cn_cpi_yoy", "cn_ppi_yoy", "cn_pmi", "cn_m2_yoy",
    "cn_margin_bal_z20", "cn_margin_bal_chg_5d", "cn_usdcny_chg_5d",
]
# NBS / PBoC publish monthly data with a lag; values become usable this many
# days after the end of the reference month (PMI comes out on the last day).
_MONTHLY_LAG_DAYS = 15
_PMI_LAG_DAYS = 1
# PBoC money supply sometimes lands after the 15th (holidays).
_M2_LAG_DAYS = 20
_MONTH_RE = re.compile(r"(\d{4})年(\d{1,2})月")
_CN_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")


def monthly_available(month_label: str, lag_days: int) -> date | None:
    """'2026年09月份' -> month end + lag_days."""
    m = _MONTH_RE.search(month_label)
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    return date(y, mo, calendar.monthrange(y, mo)[1]) + timedelta(days=lag_days)


def _series(dates, values) -> pl.DataFrame:
    return (
        pl.DataFrame({"date": list(dates), "value": list(values)},
                     schema={"date": pl.Date, "value": pl.Float64}, strict=False)
        .drop_nulls()
        .unique("date", keep="last")
        .sort("date")
    )


def _monthly(df, col: str, lag: int) -> pl.DataFrame:
    return _series([monthly_available(str(m), lag) for m in df["月份"]], df[col])


def fetch_cn_macro_raw() -> dict[str, pl.DataFrame]:
    import pandas as pd

    pd.set_option("future.infer_string", False)
    import akshare as ak

    raw: dict[str, pl.DataFrame] = {}
    lpr = ak.macro_china_lpr()
    raw["lpr_1y"] = _series(pd.to_datetime(lpr["TRADE_DATE"]).dt.date, lpr["LPR1Y"])

    rrr = ak.macro_china_reserve_requirement_ratio()
    ann = []
    for s in rrr["公布时间"].astype(str):
        m = _CN_DATE_RE.search(s)
        # Announced in the evening -> usable the next day.
        ann.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))) + timedelta(days=1) if m else None)
    raw["rrr"] = _series(ann, rrr["大型金融机构-调整后"])

    raw["cpi_yoy"] = _monthly(ak.macro_china_cpi(), "全国-同比增长", _MONTHLY_LAG_DAYS)
    raw["ppi_yoy"] = _monthly(ak.macro_china_ppi(), "当月同比增长", _MONTHLY_LAG_DAYS)
    raw["pmi"] = _monthly(ak.macro_china_pmi(), "制造业-指数", _PMI_LAG_DAYS)
    raw["m2_yoy"] = _monthly(ak.macro_china_money_supply(), "货币和准货币(M2)-同比增长", _M2_LAG_DAYS)

    # Exchange margin balances for day D are published the next morning.
    sh, sz = ak.macro_china_market_margin_sh(), ak.macro_china_market_margin_sz()
    tot = (
        pl.concat([
            _series(pd.to_datetime(sh["日期"]).dt.date, sh["融资融券余额"]),
            _series(pd.to_datetime(sz["日期"]).dt.date, sz["融资融券余额"]),
        ])
        .group_by("date").agg(pl.col("value").sum(), pl.len().alias("n"))
        .filter(pl.col("n") == 2).drop("n").sort("date")
        .with_columns(pl.col("date") + pl.duration(days=1))
    )
    raw["margin_total"] = tot

    fx = ak.currency_boc_safe()  # central parity, 09:15 Beijing, CNY per 100 USD
    raw["usdcny"] = _series(pd.to_datetime(fx["日期"]).dt.date, fx["美元"] / 100.0)
    return raw


def build_cn_macro_daily(raw: dict[str, pl.DataFrame], days: list[date] | pl.Series) -> pl.DataFrame:
    """One row per trading day with `date` + CN_MACRO_COLS; values as of that day only."""
    base = pl.DataFrame({"date": pl.Series(list(days), dtype=pl.Date)}).unique().sort("date")
    names = {
        "lpr_1y": "cn_lpr_1y", "rrr": "cn_rrr", "cpi_yoy": "cn_cpi_yoy",
        "ppi_yoy": "cn_ppi_yoy", "pmi": "cn_pmi", "m2_yoy": "cn_m2_yoy",
        "margin_total": "_margin", "usdcny": "_usdcny",
    }
    for key, col in names.items():
        s = raw.get(key)
        if s is None or s.height == 0:
            base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
            continue
        base = base.join_asof(s.sort("date").rename({"value": col}), on="date", strategy="backward")

    rrr = raw.get("rrr")
    cut_days: list[date] = []
    if rrr is not None and rrr.height > 1:
        r = rrr.sort("date").with_columns((pl.col("value").diff() < 0).alias("cut"))
        cuts = r.filter(pl.col("cut")).select("date").sort("date")
        # A cut announced on a non-trading day counts from the next trading day.
        mapped = cuts.join_asof(
            base.select(pl.col("date").alias("_td"), pl.col("date")), on="date", strategy="forward"
        )
        cut_days = mapped["_td"].drop_nulls().to_list()
    base = base.with_columns(pl.col("date").is_in(cut_days).cast(pl.Float64).alias("_cut"))

    m = pl.col("_margin")
    z = (m - m.rolling_mean(20, min_samples=10)) / m.rolling_std(20, min_samples=10)
    out = base.with_columns(
        (pl.col("cn_lpr_1y") - pl.col("cn_lpr_1y").shift(20)).alias("cn_lpr_1y_chg_20d"),
        pl.col("_cut").rolling_max(20, min_samples=1).alias("cn_rrr_cut_20d"),
        pl.when(z.is_finite()).then(z).otherwise(None).alias("cn_margin_bal_z20"),
        (m / m.shift(5) - 1.0).alias("cn_margin_bal_chg_5d"),
        (pl.col("_usdcny") / pl.col("_usdcny").shift(5) - 1.0).alias("cn_usdcny_chg_5d"),
    )
    return out.select(["date", *CN_MACRO_COLS])
