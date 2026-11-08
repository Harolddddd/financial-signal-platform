from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow.parquet as pq

from src.ingestion import news_store as ns

ET = ZoneInfo("America/New_York")


def _ts(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _scores(rows):
    return pl.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "published_at": [r[1] for r in rows],
            "url_hash": [r[2] for r in rows],
            "sentiment_pos": [r[3] for r in rows],
            "sentiment_neg": [0.1] * len(rows),
            "sentiment_neu": [0.2] * len(rows),
            "scorer": ["finbert"] * len(rows),
        }
    ).with_columns(pl.col("published_at").dt.convert_time_zone("UTC"))


def _days(*ds):
    return pl.Series("d", list(ds), dtype=pl.Date)


def test_url_hash_stable_and_int64():
    a = ns.url_hash("https://x.com/a")
    assert a == ns.url_hash("https://x.com/a")
    assert a != ns.url_hash("https://x.com/b")
    assert -(2**63) <= a < 2**63


def test_assign_trading_day_cutoff_and_weekend():
    pub = pl.Series(
        "p",
        [_ts(2026, 10, 30, 15, 59), _ts(2026, 10, 30, 16, 0), _ts(2026, 10, 31, 10, 0)],
    ).dt.convert_time_zone("UTC")
    out = ns.assign_trading_day(pub, _days(date(2026, 10, 30), date(2026, 11, 2)))
    assert out.to_list() == [date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 2)]


def test_write_scores_dedupes(tmp_path):
    df = _scores([("AAPL", _ts(2026, 10, 30, 10, 0), i, 0.5) for i in range(3)])
    assert ns.write_scores(tmp_path, df) == 3
    assert ns.write_scores(tmp_path, df) == 0
    files = list(tmp_path.glob("articles_scores_*.parquet"))
    assert len(files) == 1
    assert pl.read_parquet(files[0]).height == 3


def test_rebuild_daily_aggregates_and_keeps_old(tmp_path):
    old = pl.DataFrame(
        {
            "ticker": ["MSFT"],
            "time": [date(2026, 10, 1)],
            "avg_pos": [0.9],
            "avg_neg": [0.0],
            "avg_neu": [0.1],
            "article_count": [4],
        }
    )
    old.write_parquet(tmp_path / "sentiment_daily.parquet", compression="zstd")
    df = _scores(
        [
            ("AAPL", _ts(2026, 10, 30, 9, 0), 1, 0.2),
            ("AAPL", _ts(2026, 10, 30, 11, 0), 2, 0.6),
        ]
    )
    ns.write_scores(tmp_path, df)
    ns.rebuild_daily(tmp_path, _days(date(2026, 10, 30), date(2026, 11, 2)), date(2026, 10, 15))
    out = ns.load_daily(tmp_path)
    assert out.filter(pl.col("ticker") == "MSFT").height == 1
    row = out.filter(pl.col("ticker") == "AAPL").row(0, named=True)
    assert row["article_count"] == 2
    assert abs(row["avg_pos"] - 0.4) < 1e-5
    assert row["time"] == date(2026, 10, 30)


def _old_daily(tmp_path, rows):
    pl.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "avg_pos": [r[2] for r in rows],
            "avg_neg": [0.0] * len(rows),
            "avg_neu": [0.1] * len(rows),
            "article_count": [7] * len(rows),
        }
    ).write_parquet(tmp_path / "sentiment_daily.parquet", compression="zstd")


def test_rebuild_with_early_since_keeps_rows_outside_score_coverage(tmp_path):
    # permanent-tier rows from months whose score files were pruned long ago
    _old_daily(
        tmp_path,
        [
            ("AAPL", date(2026, 3, 2), 0.9),
            ("AAPL", date(2026, 10, 30), 0.99),  # recomputed below -> replaced
            ("MSFT", date(2026, 6, 1), 0.8),  # ticker absent from scores
        ],
    )
    ns.write_scores(tmp_path, _scores([("AAPL", _ts(2026, 10, 30, 9, 0), 1, 0.2)]))
    ns.rebuild_daily(tmp_path, _days(date(2026, 10, 30), date(2026, 11, 2)), date(2026, 1, 1))
    out = ns.load_daily(tmp_path).with_columns(pl.col("ticker").cast(pl.Utf8))
    keys = {(r["ticker"], r["time"]): r for r in out.iter_rows(named=True)}
    assert set(keys) == {
        ("AAPL", date(2026, 3, 2)),
        ("AAPL", date(2026, 10, 30)),
        ("MSFT", date(2026, 6, 1)),
    }
    assert abs(keys[("AAPL", date(2026, 3, 2))]["avg_pos"] - 0.9) < 1e-5
    assert abs(keys[("MSFT", date(2026, 6, 1))]["avg_pos"] - 0.8) < 1e-5
    replaced = keys[("AAPL", date(2026, 10, 30))]
    assert abs(replaced["avg_pos"] - 0.2) < 1e-5
    assert replaced["article_count"] == 1


def test_rebuild_with_no_score_files_keeps_all_rows(tmp_path):
    _old_daily(tmp_path, [("MSFT", date(2026, 10, 30), 0.8)])
    ns.rebuild_daily(tmp_path, _days(date(2026, 10, 30)), date(2026, 1, 1))
    assert ns.load_daily(tmp_path).height == 1


def test_load_daily_missing_returns_empty_schema(tmp_path):
    out = ns.load_daily(tmp_path)
    assert out.height == 0
    assert out.columns == ["ticker", "time", "avg_pos", "avg_neg", "avg_neu", "article_count"]


def test_prune_keeps_recent_months(tmp_path):
    ns.write_scores(tmp_path, _scores([("AAPL", _ts(2026, 6, 10, 10, 0), 1, 0.5)]))
    ns.write_scores(tmp_path, _scores([("AAPL", _ts(2026, 10, 10, 10, 0), 2, 0.5)]))
    ns.prune(tmp_path, date(2026, 11, 6), keep_days=60)
    names = sorted(p.name for p in tmp_path.glob("articles_scores_*.parquet"))
    assert names == ["articles_scores_2026-10.parquet"]


def test_headlines_rolls_off_after_14_days(tmp_path):
    def rows(h, d):
        return pl.DataFrame(
            {
                "url_hash": [h],
                "ticker": ["AAPL"],
                "published_at": [datetime(2026, 11, d, 12, 0, tzinfo=ET)],
                "headline": [f"h{h}"],
            }
        ).with_columns(pl.col("published_at").dt.convert_time_zone("UTC"))

    ns.write_headlines(tmp_path, rows(1, 1), date(2026, 11, 5))
    ns.write_headlines(tmp_path, rows(2, 20), date(2026, 11, 20))
    out = pl.read_parquet(tmp_path / "headlines_recent.parquet")
    assert out["url_hash"].to_list() == [2]


def test_files_use_zstd(tmp_path):
    ns.write_scores(tmp_path, _scores([("AAPL", _ts(2026, 10, 30, 10, 0), 1, 0.5)]))
    ns.rebuild_daily(tmp_path, _days(date(2026, 10, 30)), date(2026, 10, 1))
    for p in (tmp_path / "articles_scores_2026-10.parquet", tmp_path / "sentiment_daily.parquet"):
        assert pq.ParquetFile(p).metadata.row_group(0).column(0).compression == "ZSTD"


def test_china_cutoff_rolls_after_1500_shanghai():
    from datetime import time, timezone
    days = _days(date(2026, 11, 9), date(2026, 11, 10))
    ts = pl.Series([datetime(2026, 11, 9, 6, 59, tzinfo=timezone.utc),   # 14:59 CST
                    datetime(2026, 11, 9, 7, 0, tzinfo=timezone.utc)])   # 15:00 CST
    out = ns.assign_trading_day(ts, days, tz="Asia/Shanghai", cutoff=time(15, 0))
    assert out.to_list() == [date(2026, 11, 9), date(2026, 11, 10)]


def test_rebuild_daily_china_cutoff(tmp_path):
    from datetime import time, timezone
    sh = ZoneInfo("Asia/Shanghai")
    ns.write_scores(tmp_path, _scores([
        ("600000.SS", datetime(2026, 11, 9, 15, 30, tzinfo=sh), 1, 0.9),
    ]))
    ns.rebuild_daily(tmp_path, _days(date(2026, 11, 9), date(2026, 11, 10)), since=date(2026, 11, 1),
                     tz="Asia/Shanghai", cutoff=time(15, 0))
    d = ns.load_daily(tmp_path)
    assert d["time"].to_list() == [date(2026, 11, 10)]
