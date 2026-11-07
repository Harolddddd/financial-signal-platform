from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from scripts import refresh_macro
from src.ingestion import macro_ingest

NY = ZoneInfo("America/New_York")
N = 30
DAYS = [date(2024, 1, 1) + timedelta(days=i) for i in range(N)]


def _setup(tmp_path, monkeypatch):
    ohlcv = tmp_path / "raw" / "ohlcv"
    ohlcv.mkdir(parents=True)
    pl.DataFrame(
        {"time": [datetime(d.year, d.month, d.day, tzinfo=timezone.utc) for d in DAYS]}
    ).write_parquet(ohlcv / "SPY.parquet")
    monkeypatch.setattr(refresh_macro.settings, "FRED_KEY", "k")
    monkeypatch.setattr(refresh_macro, "get_market", lambda m: SimpleNamespace(data_root=tmp_path))
    series = pl.DataFrame({"date": DAYS, "value": [float(i) for i in range(N)]})
    monkeypatch.setattr(refresh_macro, "fetch_fred", lambda sid, start, key: series)
    return tmp_path / "macro"


def _raising_classifier(sentences):
    raise RuntimeError("transformers model load failed")


def _statement():
    return [(datetime(2024, 1, 10, 14, 0, tzinfo=NY),
             "The Committee decided to raise the target range for the federal funds rate.")]


def test_fomc_classifier_failure_still_writes_fred_columns(tmp_path, monkeypatch, caplog):
    macro_dir = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(refresh_macro, "fetch_fomc_statements", lambda since: _statement())
    real = macro_ingest.score_fomc_statements
    monkeypatch.setattr(
        refresh_macro, "score_fomc_statements",
        lambda st: real(st, classify=_raising_classifier),
    )
    with caplog.at_level("WARNING"):
        refresh_macro.main()
    out = pl.read_parquet(macro_dir / "macro_daily.parquet")
    assert out.height == N
    assert out["macro_dgs2"].drop_nulls().len() > 0
    assert "fomc_tone" in out.columns
    assert "FOMC" in caplog.text


def test_fomc_fetch_failure_uses_cached_scores(tmp_path, monkeypatch, caplog):
    macro_dir = _setup(tmp_path, monkeypatch)
    macro_dir.mkdir()
    cached = pl.DataFrame(
        {"release_ts": [datetime(2024, 1, 10, 14, 0, tzinfo=NY)], "tone": [0.5]},
        schema={"release_ts": pl.Datetime("us", "America/New_York"), "tone": pl.Float64},
    )
    cached.write_parquet(macro_dir / "fomc_scores.parquet")

    def boom(since):
        raise RuntimeError("fed site down")

    monkeypatch.setattr(refresh_macro, "fetch_fomc_statements", boom)
    with caplog.at_level("WARNING"):
        refresh_macro.main()
    out = pl.read_parquet(macro_dir / "macro_daily.parquet")
    assert out.height == N
    assert out["macro_dgs2"].drop_nulls().len() > 0
    assert 0.5 in out["fomc_tone"].drop_nulls().to_list()
    # cached scores are left intact
    assert pl.read_parquet(macro_dir / "fomc_scores.parquet").equals(cached)
    assert "FOMC" in caplog.text
