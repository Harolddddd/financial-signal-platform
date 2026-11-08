from datetime import date, datetime, timezone

import polars as pl
import pytest

from src.features.news_macro_features import (
    MACRO_COLS,
    NEWS_COLS,
    OVERLAY_COLS,
    add_macro_features,
    add_news_features,
)


def _frame(n: int = 5, ticker: str = "AAPL") -> pl.DataFrame:
    return pl.DataFrame({
        "time": [datetime(2024, 1, 2 + i, tzinfo=timezone.utc) for i in range(n)],
        "ticker": [ticker] * n,
        "close": [100.0 + i for i in range(n)],
    })


def _news(rows):
    return pl.DataFrame(
        rows,
        schema={"ticker": pl.Utf8, "time": pl.Date, "avg_pos": pl.Float64,
                "avg_neg": pl.Float64, "avg_neu": pl.Float64, "article_count": pl.Int64},
        orient="row",
    )


def test_no_news_gives_neutral_columns_same_rows():
    df = _frame()
    out = add_news_features(df, None)
    assert out.height == df.height
    for c in NEWS_COLS:
        assert c in out.columns
    assert out.null_count().sum_horizontal().item() == 0
    assert out["news_has_3d"].dtype == pl.Boolean
    assert not out["news_has_3d"].any()
    assert out["news_count_1d"].sum() == 0


def test_news_net_3d_weighted():
    df = _frame(3)
    news = _news([
        ("AAPL", date(2024, 1, 2), 1.0, 0.0, 0.0, 1),
        ("AAPL", date(2024, 1, 3), 0.0, 0.0, 1.0, 3),
    ])
    out = add_news_features(df, news)
    assert out["news_net_3d"][1] == pytest.approx(0.25)
    assert out["news_count_3d"][2] == 4
    assert out["news_has_3d"].to_list() == [True, True, True]
    assert out.null_count().sum_horizontal().item() == 0


def test_news_join_is_per_ticker_and_keeps_order():
    df = _frame(3)
    news = _news([("MSFT", date(2024, 1, 2), 1.0, 0.0, 0.0, 5)])
    out = add_news_features(df, news)
    assert out["news_count_1d"].sum() == 0
    assert out["time"].to_list() == df["time"].to_list()


def test_news_z_scores_use_only_past_rows():
    n = 30
    df = _frame(n)
    rows = [("AAPL", df["time"][i].date(), 0.5, 0.1, 0.4, i % 3 + 1) for i in range(n)]
    out = add_news_features(df, _news(rows))
    assert out["news_vol_z20"][:9].to_list() == [0.0] * 9
    assert out["news_net_z60"][:19].to_list() == [0.0] * 19
    rows2 = list(rows)
    rows2[-1] = (*rows2[-1][:5], 500)
    out2 = add_news_features(df, _news(rows2))
    assert out2["news_vol_z20"][:-1].to_list() == out["news_vol_z20"][:-1].to_list()
    assert out2["news_net_z60"][:-1].to_list() == out["news_net_z60"][:-1].to_list()
    assert out2.null_count().sum_horizontal().item() == 0


def _macro():
    d = {"date": [date(2024, 1, 3), date(2024, 1, 4)]}
    for c in MACRO_COLS:
        d[c] = [1.5, 2.5] if c != "days_since_fomc" else [3, 4]
    return pl.DataFrame(d)


def test_macro_none_gives_neutral_values():
    out = add_macro_features(_frame(), None)
    assert out.height == 5
    for c in MACRO_COLS:
        assert c in out.columns
    assert out["days_since_fomc"].to_list() == [999] * 5
    assert out["macro_dff"].to_list() == [0.0] * 5


def test_macro_join_fills_missing_dates():
    out = add_macro_features(_frame(), _macro())
    assert out.null_count().sum_horizontal().item() == 0
    assert out["macro_dff"].to_list() == [0.0, 1.5, 2.5, 0.0, 0.0]
    assert out["days_since_fomc"].to_list() == [999, 3, 4, 999, 999]
    assert "date" not in out.columns


def test_macro_fills_null_first_row():
    m = _macro().with_columns(pl.when(pl.col("date") == date(2024, 1, 3)).then(None).otherwise(pl.col("macro_dgs2")).alias("macro_dgs2"))
    out = add_macro_features(_frame(), m)
    assert out["macro_dgs2"].null_count() == 0


def test_overlay_cols_is_news_plus_macro():
    assert OVERLAY_COLS == NEWS_COLS + MACRO_COLS
    assert len(MACRO_COLS) == 12


def test_load_us_news_macro_warns_when_files_missing(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from src.features import news_macro_features as nmf

    monkeypatch.setattr(nmf, "get_market", lambda m: SimpleNamespace(data_root=tmp_path))
    with caplog.at_level("WARNING"):
        news, macro = nmf.load_us_news_macro()
    assert news.height == 0 and macro.height == 0
    assert "macro_daily.parquet" in caplog.text
    assert "sentiment_daily.parquet" in caplog.text
    # neutral columns are still produced
    out = add_macro_features(add_news_features(_frame(), news), macro)
    assert set(OVERLAY_COLS) <= set(out.columns)
    assert out.select(OVERLAY_COLS).null_count().sum_horizontal().item() == 0


def test_overlay_cols_by_market():
    from src.features.news_macro_features import MACRO_COLS, NEWS_COLS, OVERLAY_COLS, overlay_cols
    from src.ingestion.cn_macro_ingest import CN_MACRO_COLS
    assert overlay_cols("us") == OVERLAY_COLS == NEWS_COLS + MACRO_COLS
    assert overlay_cols("china") == NEWS_COLS + CN_MACRO_COLS


def test_china_macro_neutral_when_missing():
    from datetime import datetime
    from src.features.news_macro_features import add_macro_features
    from src.ingestion.cn_macro_ingest import CN_MACRO_COLS
    df = pl.DataFrame({"time": [datetime(2026, 1, 5), datetime(2026, 1, 6)], "ticker": ["A", "A"]})
    out = add_macro_features(df, None, cols=CN_MACRO_COLS)
    assert all(out[c].to_list() == [0.0, 0.0] for c in CN_MACRO_COLS)
    macro = pl.DataFrame({"date": [datetime(2026, 1, 5).date()], **{c: [1.5] for c in CN_MACRO_COLS}})
    out = add_macro_features(df, macro, cols=CN_MACRO_COLS)
    assert out["cn_pmi"].to_list() == [1.5, 0.0]
