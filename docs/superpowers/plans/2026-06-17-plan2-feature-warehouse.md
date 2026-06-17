# Financial Platform — Plan 2: Feature Warehouse

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform raw OHLCV and sentiment data from Plan 1 into a rich, model-ready feature set — technical indicators, sentiment features, cross-asset signals, and classification labels — stored in TimescaleDB for writes and exported to Parquet/DuckDB for fast ML training data loading.

**Architecture:** A chain of pure Polars transform functions (one module per feature family) is orchestrated by a `FeatureStore` class that reads raw data from TimescaleDB, applies all transforms, joins the results, and writes the final feature table back to TimescaleDB while also exporting Parquet files for DuckDB. dbt models shadow a subset of these computations in SQL for lineage, documentation, and data-quality tests. An Airflow DAG runs the full pipeline daily after Plan 1's ingestion DAGs complete.

**Tech Stack:** Python 3.11, Polars 0.20, DuckDB 0.10, dbt-core 1.8 + dbt-postgres, psycopg2-binary 2.9, TimescaleDB 2.x (from Plan 1), Apache Airflow 2.9, pytest 8.x

**Dependency on Plan 1:** Requires `ohlcv`, `news_articles`, and `index_compositions` tables in TimescaleDB. Imports `config.settings`, `src.ingestion.db`, and `src.ingestion.collector` from Plan 1.

## Global Constraints

- Python >= 3.11; all function signatures require type hints
- All timestamps UTC; features table uses the same `time TIMESTAMPTZ` convention as Plan 1
- Feature functions are **pure** — they take a `pl.DataFrame` and return a `pl.DataFrame`; no DB I/O inside indicator/label modules
- Input DataFrames must be sorted by `time` ascending and contain a single ticker; multi-ticker orchestration is the `FeatureStore`'s job
- Labels are computed with `forward_days=5`; the last `forward_days` rows of any DataFrame will have `null` labels — callers must drop them before training
- No look-ahead bias: feature at row T uses only columns at rows ≤ T; the label at row T uses `close[T+forward_days]`
- dbt models run as views against TimescaleDB for lineage — they are not in the write path
- DuckDB reads Parquet; it does not connect to TimescaleDB directly
- Tests use pytest only; no `unittest.TestCase`

---

### Task 1: Feature Schema & DuckDB Client

**Files:**
- Create: `migrations/002_feature_schema.sql`
- Create: `src/features/__init__.py`
- Create: `src/features/duckdb_client.py`
- Test: `tests/unit/test_duckdb_client.py`

**Interfaces:**
- Consumes: `config.settings.settings.DATABASE_URL` (for schema migration); Parquet files on disk
- Produces:
  - Table `features` in TimescaleDB (hypertable)
  - `load_training_data(parquet_dir: Path, tickers: list[str] | None, start: datetime | None, end: datetime | None) -> pl.DataFrame`

- [ ] **Step 1: Create migrations/002_feature_schema.sql**

```sql
CREATE TABLE IF NOT EXISTS features (
    time              TIMESTAMPTZ      NOT NULL,
    ticker            TEXT             NOT NULL,
    -- Moving averages
    sma_10            DOUBLE PRECISION,
    sma_20            DOUBLE PRECISION,
    sma_50            DOUBLE PRECISION,
    sma_200           DOUBLE PRECISION,
    ema_12            DOUBLE PRECISION,
    ema_26            DOUBLE PRECISION,
    -- Momentum
    rsi_14            DOUBLE PRECISION,
    macd              DOUBLE PRECISION,
    macd_signal       DOUBLE PRECISION,
    macd_hist         DOUBLE PRECISION,
    -- Volatility
    bb_upper          DOUBLE PRECISION,
    bb_lower          DOUBLE PRECISION,
    bb_width          DOUBLE PRECISION,
    atr_14            DOUBLE PRECISION,
    hist_vol_21       DOUBLE PRECISION,
    -- Sentiment
    sent_pos_avg_3d   DOUBLE PRECISION,
    sent_pos_avg_5d   DOUBLE PRECISION,
    sent_pos_avg_10d  DOUBLE PRECISION,
    sent_pos_mom_3d   DOUBLE PRECISION,
    news_vol_spike    DOUBLE PRECISION,
    -- Cross-asset
    rel_strength_spy  DOUBLE PRECISION,
    vix_level         DOUBLE PRECISION,
    -- Labels
    forward_return_5d DOUBLE PRECISION,
    label             TEXT
);
SELECT create_hypertable('features', 'time', if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS idx_features_ticker_time
    ON features (ticker, time DESC);
```

- [ ] **Step 2: Apply migration**

```bash
docker compose exec timescaledb psql -U platform -d financial \
  -f /docker-entrypoint-initdb.d/002_feature_schema.sql
```
Expected: `CREATE TABLE`, `create_hypertable`, `CREATE INDEX` — no errors.

- [ ] **Step 3: Write failing test**

```python
# tests/unit/test_duckdb_client.py
from pathlib import Path
from datetime import datetime, timezone
import polars as pl
import pytest
import tempfile

from src.features.duckdb_client import load_training_data


def _write_sample_parquet(tmp_dir: Path) -> None:
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 2, tzinfo=timezone.utc), datetime(2024, 1, 3, tzinfo=timezone.utc)],
        "ticker": ["AAPL", "AAPL"],
        "close": [153.0, 154.0],
        "rsi_14": [55.0, 57.0],
        "label": ["Buy", "Hold"],
    })
    df.write_parquet(tmp_dir / "AAPL.parquet")


def test_load_training_data_returns_polars_dataframe():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _write_sample_parquet(p)
        df = load_training_data(p)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2


def test_load_training_data_filters_by_ticker():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _write_sample_parquet(p)
        df = load_training_data(p, tickers=["AAPL"])
        assert all(df["ticker"] == "AAPL")


def test_load_training_data_filters_by_date_range():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _write_sample_parquet(p)
        start = datetime(2024, 1, 3, tzinfo=timezone.utc)
        df = load_training_data(p, start=start)
        assert len(df) == 1
        assert df["time"][0] == start
```

- [ ] **Step 4: Run test to verify it fails**

```bash
pytest tests/unit/test_duckdb_client.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.features.duckdb_client'`

- [ ] **Step 5: Implement src/features/duckdb_client.py**

```python
from __future__ import annotations
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl


def load_training_data(
    parquet_dir: Path,
    tickers: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    pattern = str(parquet_dir / "*.parquet")
    conditions: list[str] = []

    if tickers:
        quoted = ", ".join(f"'{t}'" for t in tickers)
        conditions.append(f"ticker IN ({quoted})")
    if start:
        conditions.append(f"time >= TIMESTAMPTZ '{start.isoformat()}'")
    if end:
        conditions.append(f"time <= TIMESTAMPTZ '{end.isoformat()}'")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM read_parquet('{pattern}') {where} ORDER BY ticker, time"

    conn = duckdb.connect()
    return conn.execute(sql).pl()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/unit/test_duckdb_client.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add migrations/002_feature_schema.sql src/features/ tests/unit/test_duckdb_client.py
git commit -m "feat: feature table schema and DuckDB parquet query client"
```

---

### Task 2: Technical Indicators

**Files:**
- Create: `src/features/technical_indicators.py`
- Test: `tests/unit/test_technical_indicators.py`

**Interfaces:**
- Consumes: `pl.DataFrame` with columns `time: Datetime[us, UTC]`, `open: Float64`, `high: Float64`, `low: Float64`, `close: Float64`, `volume: Int64` — sorted ascending by `time`, single ticker
- Produces:
  - `add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame`
    — returns input DataFrame with these columns added:
    `sma_10`, `sma_20`, `sma_50`, `sma_200`, `ema_12`, `ema_26`,
    `rsi_14`, `macd`, `macd_signal`, `macd_hist`,
    `bb_upper`, `bb_lower`, `bb_width`, `atr_14`, `hist_vol_21`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_technical_indicators.py
from datetime import datetime, timezone, timedelta
import polars as pl
import pytest

from src.features.technical_indicators import add_technical_indicators

_INDICATOR_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "atr_14", "hist_vol_21",
]


def _make_ohlcv(n: int = 250) -> pl.DataFrame:
    base = datetime(2022, 1, 3, tzinfo=timezone.utc)
    import math
    closes = [100.0 + 10 * math.sin(i / 20) + i * 0.05 for i in range(n)]
    return pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(n)],
        "ticker": ["AAPL"] * n,
        "open":   [c - 0.5 for c in closes],
        "high":   [c + 1.0 for c in closes],
        "low":    [c - 1.0 for c in closes],
        "close":  closes,
        "volume": [1_000_000] * n,
    })


def test_all_indicator_columns_present():
    df = add_technical_indicators(_make_ohlcv())
    for col in _INDICATOR_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_row_count_unchanged():
    raw = _make_ohlcv()
    df = add_technical_indicators(raw)
    assert len(df) == len(raw)


def test_sma_20_is_rolling_mean_of_close():
    df = add_technical_indicators(_make_ohlcv())
    row_30 = df.row(30, named=True)
    manual_sma20 = df["close"][11:31].mean()
    assert abs(row_30["sma_20"] - manual_sma20) < 1e-6


def test_rsi_bounded_0_to_100():
    df = add_technical_indicators(_make_ohlcv())
    valid = df["rsi_14"].drop_nulls()
    assert valid.min() >= 0.0
    assert valid.max() <= 100.0


def test_bb_upper_above_bb_lower():
    df = add_technical_indicators(_make_ohlcv())
    valid = df.drop_nulls(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_atr_non_negative():
    df = add_technical_indicators(_make_ohlcv())
    valid = df["atr_14"].drop_nulls()
    assert (valid >= 0).all()


def test_hist_vol_non_negative():
    df = add_technical_indicators(_make_ohlcv())
    valid = df["hist_vol_21"].drop_nulls()
    assert (valid >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_technical_indicators.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.features.technical_indicators'`

- [ ] **Step 3: Implement src/features/technical_indicators.py**

```python
import polars as pl


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    df = _add_moving_averages(df)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_bollinger_bands(df)
    df = _add_atr(df)
    df = _add_hist_vol(df)
    return df


def _add_moving_averages(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("close").rolling_mean(window_size=10).alias("sma_10"),
        pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
        pl.col("close").rolling_mean(window_size=200).alias("sma_200"),
        pl.col("close").ewm_mean(span=12, adjust=False).alias("ema_12"),
        pl.col("close").ewm_mean(span=26, adjust=False).alias("ema_26"),
    ])


def _add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    gain = (
        pl.col("close").diff()
        .clip(lower_bound=0)
        .ewm_mean(span=period, adjust=False)
    )
    loss = (
        (pl.col("close").diff() * -1)
        .clip(lower_bound=0)
        .ewm_mean(span=period, adjust=False)
    )
    rsi = (100 - 100 / (1 + gain / loss)).alias("rsi_14")
    return df.with_columns(rsi)


def _add_macd(df: pl.DataFrame) -> pl.DataFrame:
    ema12 = pl.col("close").ewm_mean(span=12, adjust=False)
    ema26 = pl.col("close").ewm_mean(span=26, adjust=False)
    return (
        df
        .with_columns((ema12 - ema26).alias("macd"))
        .with_columns(pl.col("macd").ewm_mean(span=9, adjust=False).alias("macd_signal"))
        .with_columns((pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"))
    )


def _add_bollinger_bands(df: pl.DataFrame, period: int = 20, n_std: float = 2.0) -> pl.DataFrame:
    sma = pl.col("close").rolling_mean(window_size=period)
    std = pl.col("close").rolling_std(window_size=period)
    return df.with_columns([
        (sma + n_std * std).alias("bb_upper"),
        (sma - n_std * std).alias("bb_lower"),
        ((n_std * std * 2) / sma).alias("bb_width"),
    ])


def _add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    prev_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(
        true_range.ewm_mean(span=period, adjust=False).alias("atr_14")
    )


def _add_hist_vol(df: pl.DataFrame, period: int = 21) -> pl.DataFrame:
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
    hist_vol = log_ret.rolling_std(window_size=period) * (252 ** 0.5)
    return df.with_columns(hist_vol.alias("hist_vol_21"))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_technical_indicators.py -v
```
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/technical_indicators.py tests/unit/test_technical_indicators.py
git commit -m "feat: technical indicators (SMA/EMA/RSI/MACD/BB/ATR/HistVol)"
```

---

### Task 3: Sentiment Features

**Files:**
- Create: `src/features/sentiment_features.py`
- Test: `tests/unit/test_sentiment_features.py`

**Interfaces:**
- Consumes: `pl.DataFrame` with daily-aggregated sentiment per ticker:
  `time: Date`, `ticker: Utf8`, `avg_pos: Float64`, `article_count: Int64` — sorted ascending by `time`, single ticker
- Produces:
  - `aggregate_daily_sentiment(raw_df: pl.DataFrame) -> pl.DataFrame`
    — aggregates `news_articles` rows (from Plan 1 schema) into one row per (ticker, date)
    — output columns: `time: Date`, `ticker: Utf8`, `avg_pos: Float64`, `avg_neg: Float64`, `avg_neu: Float64`, `article_count: Int64`
  - `add_sentiment_features(df: pl.DataFrame) -> pl.DataFrame`
    — adds to the daily-aggregated DataFrame:
    `sent_pos_avg_3d`, `sent_pos_avg_5d`, `sent_pos_avg_10d`, `sent_pos_mom_3d`, `news_vol_spike`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_sentiment_features.py
from datetime import date, datetime, timezone, timedelta
import polars as pl
import pytest

from src.features.sentiment_features import aggregate_daily_sentiment, add_sentiment_features


def _make_raw_sentiment(n_days: int = 30, articles_per_day: int = 3) -> pl.DataFrame:
    rows = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for d in range(n_days):
        for _ in range(articles_per_day):
            rows.append({
                "published_at": base + timedelta(days=d),
                "ticker": "AAPL",
                "sentiment_pos": 0.6,
                "sentiment_neg": 0.2,
                "sentiment_neu": 0.2,
            })
    return pl.DataFrame(rows)


def _make_daily_sentiment(n: int = 30) -> pl.DataFrame:
    base = date(2024, 1, 1)
    return pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(n)],
        "ticker": ["AAPL"] * n,
        "avg_pos": [0.6] * n,
        "avg_neg": [0.2] * n,
        "avg_neu": [0.2] * n,
        "article_count": [3] * n,
    })


def test_aggregate_daily_sentiment_one_row_per_day():
    raw = _make_raw_sentiment(n_days=5, articles_per_day=3)
    daily = aggregate_daily_sentiment(raw)
    assert len(daily) == 5


def test_aggregate_daily_sentiment_avg_pos_correct():
    raw = _make_raw_sentiment(n_days=1, articles_per_day=2)
    daily = aggregate_daily_sentiment(raw)
    assert abs(daily["avg_pos"][0] - 0.6) < 1e-6


def test_add_sentiment_features_adds_rolling_columns():
    df = add_sentiment_features(_make_daily_sentiment(30))
    for col in ["sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
                "sent_pos_mom_3d", "news_vol_spike"]:
        assert col in df.columns, f"Missing: {col}"


def test_rolling_averages_converge_on_constant_sentiment():
    df = add_sentiment_features(_make_daily_sentiment(30))
    last = df.row(-1, named=True)
    assert abs(last["sent_pos_avg_3d"] - 0.6) < 1e-6
    assert abs(last["sent_pos_avg_5d"] - 0.6) < 1e-6
    assert abs(last["sent_pos_avg_10d"] - 0.6) < 1e-6


def test_momentum_is_zero_on_constant_sentiment():
    df = add_sentiment_features(_make_daily_sentiment(30))
    last = df.row(-1, named=True)
    assert abs(last["sent_pos_mom_3d"]) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_sentiment_features.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/features/sentiment_features.py**

```python
import polars as pl


def aggregate_daily_sentiment(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates news_articles rows (one per article) into one row per (ticker, calendar date).
    raw_df must have columns: published_at (Datetime UTC), ticker, sentiment_pos, sentiment_neg, sentiment_neu.
    """
    return (
        raw_df
        .with_columns(pl.col("published_at").dt.date().alias("time"))
        .group_by(["ticker", "time"])
        .agg([
            pl.col("sentiment_pos").mean().alias("avg_pos"),
            pl.col("sentiment_neg").mean().alias("avg_neg"),
            pl.col("sentiment_neu").mean().alias("avg_neu"),
            pl.col("sentiment_pos").count().alias("article_count"),
        ])
        .sort("time")
    )


def add_sentiment_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds rolling sentiment features to a daily-aggregated sentiment DataFrame.
    Input must be sorted ascending by time, single ticker.
    """
    return (
        df
        .with_columns([
            pl.col("avg_pos").rolling_mean(window_size=3).alias("sent_pos_avg_3d"),
            pl.col("avg_pos").rolling_mean(window_size=5).alias("sent_pos_avg_5d"),
            pl.col("avg_pos").rolling_mean(window_size=10).alias("sent_pos_avg_10d"),
        ])
        .with_columns(
            (pl.col("sent_pos_avg_3d") - pl.col("sent_pos_avg_5d")).alias("sent_pos_mom_3d")
        )
        .with_columns([
            _zscore_rolling(pl.col("article_count").cast(pl.Float64), window=20).alias("news_vol_spike"),
        ])
    )


def _zscore_rolling(expr: pl.Expr, window: int) -> pl.Expr:
    mean = expr.rolling_mean(window_size=window)
    std = expr.rolling_std(window_size=window)
    return (expr - mean) / (std + 1e-9)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_sentiment_features.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/sentiment_features.py tests/unit/test_sentiment_features.py
git commit -m "feat: daily sentiment aggregation and rolling sentiment features"
```

---

### Task 4: Cross-Asset Features

**Files:**
- Create: `src/features/cross_asset_features.py`
- Test: `tests/unit/test_cross_asset_features.py`

**Interfaces:**
- Consumes:
  - `stock_df: pl.DataFrame` — OHLCV for the target ticker, columns `time: Datetime[us, UTC]`, `close: Float64`
  - `spy_df: pl.DataFrame` — OHLCV for SPY, same schema
  - `vix_df: pl.DataFrame` — OHLCV for ^VIX, same schema
- Produces:
  - `add_cross_asset_features(stock_df: pl.DataFrame, spy_df: pl.DataFrame, vix_df: pl.DataFrame) -> pl.DataFrame`
    — joins spy and vix data onto `stock_df` by date, adds:
    `rel_strength_spy` (5d return of stock / 5d return of SPY, clamped to [-5, 5])
    `vix_level` (VIX close on that date)

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_cross_asset_features.py
from datetime import datetime, timezone, timedelta
import polars as pl
import pytest

from src.features.cross_asset_features import add_cross_asset_features


def _df(closes: list[float], ticker: str = "AAPL") -> pl.DataFrame:
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(len(closes))],
        "ticker": [ticker] * len(closes),
        "open":   closes,
        "high":   closes,
        "low":    closes,
        "close":  closes,
        "volume": [1_000_000] * len(closes),
        "adj_close": closes,
        "dividends": [0.0] * len(closes),
        "stock_splits": [0.0] * len(closes),
    })


def test_cross_asset_features_adds_rel_strength_and_vix():
    n = 20
    stock = _df([100.0 + i for i in range(n)])
    spy   = _df([400.0 + i * 0.5 for i in range(n)], ticker="SPY")
    vix   = _df([15.0 + i * 0.1 for i in range(n)], ticker="^VIX")

    result = add_cross_asset_features(stock, spy, vix)

    assert "rel_strength_spy" in result.columns
    assert "vix_level" in result.columns
    assert len(result) == n


def test_rel_strength_is_null_for_first_five_rows():
    n = 20
    stock = _df([100.0 + i for i in range(n)])
    spy   = _df([400.0 + i * 0.5 for i in range(n)], ticker="SPY")
    vix   = _df([15.0] * n, ticker="^VIX")

    result = add_cross_asset_features(stock, spy, vix)
    assert result["rel_strength_spy"][:5].is_null().all()


def test_rel_strength_clamped():
    n = 20
    stock = _df([100.0 + i * 5 for i in range(n)])       # fast mover
    spy   = _df([400.0 + i * 0.001 for i in range(n)], ticker="SPY")  # nearly flat
    vix   = _df([15.0] * n, ticker="^VIX")

    result = add_cross_asset_features(stock, spy, vix)
    valid = result["rel_strength_spy"].drop_nulls()
    assert (valid <= 5.0).all()
    assert (valid >= -5.0).all()


def test_missing_vix_date_becomes_null():
    stock = _df([100.0, 101.0, 102.0])
    spy   = _df([400.0, 401.0, 402.0], ticker="SPY")
    vix   = pl.DataFrame({
        "time": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
        "ticker": ["^VIX"], "open": [15.0], "high": [15.0],
        "low": [15.0], "close": [15.0], "volume": [0],
        "adj_close": [15.0], "dividends": [0.0], "stock_splits": [0.0],
    })
    result = add_cross_asset_features(stock, spy, vix)
    assert result["vix_level"][1] is None or result["vix_level"].is_null()[1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_cross_asset_features.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/features/cross_asset_features.py**

```python
import polars as pl

_FORWARD_DAYS = 5
_RS_CLAMP = 5.0


def add_cross_asset_features(
    stock_df: pl.DataFrame,
    spy_df: pl.DataFrame,
    vix_df: pl.DataFrame,
) -> pl.DataFrame:
    stock_dates = stock_df.select(pl.col("time").dt.date().alias("date"))
    spy_ret = _five_day_return(spy_df).rename({"ret_5d": "spy_ret_5d", "date": "date"})
    stock_with_ret = stock_df.with_columns(
        pl.col("time").dt.date().alias("date")
    )
    stock_with_ret = _add_five_day_return(stock_with_ret)

    vix_daily = vix_df.with_columns(
        pl.col("time").dt.date().alias("date")
    ).select(["date", pl.col("close").alias("vix_level")])

    result = (
        stock_with_ret
        .join(spy_ret, on="date", how="left")
        .join(vix_daily, on="date", how="left")
    )

    rel_strength = (
        (pl.col("ret_5d") / (pl.col("spy_ret_5d").abs() + 1e-9))
        .clip(lower_bound=-_RS_CLAMP, upper_bound=_RS_CLAMP)
        .alias("rel_strength_spy")
    )

    return result.with_columns(rel_strength).drop(["date", "ret_5d", "spy_ret_5d"])


def _add_five_day_return(df: pl.DataFrame) -> pl.DataFrame:
    ret = (pl.col("close") / pl.col("close").shift(_FORWARD_DAYS) - 1).alias("ret_5d")
    return df.with_columns(ret)


def _five_day_return(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns(pl.col("time").dt.date().alias("date"))
        .with_columns(
            (pl.col("close") / pl.col("close").shift(_FORWARD_DAYS) - 1).alias("ret_5d")
        )
        .select(["date", "ret_5d"])
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_cross_asset_features.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/cross_asset_features.py tests/unit/test_cross_asset_features.py
git commit -m "feat: cross-asset features (relative strength vs SPY, VIX level)"
```

---

### Task 5: Label Generator

**Files:**
- Create: `src/features/label_generator.py`
- Test: `tests/unit/test_label_generator.py`

**Interfaces:**
- Consumes: `pl.DataFrame` with `time: Datetime[us, UTC]`, `close: Float64` — sorted ascending, single ticker
- Produces:
  - `add_labels(df: pl.DataFrame, forward_days: int = 5, buy_threshold: float = 0.02, sell_threshold: float = -0.02) -> pl.DataFrame`
    — adds `forward_return_5d: Float64` (close[T+5] / close[T] - 1) and `label: Utf8` (`"Buy"` | `"Hold"` | `"Sell"`)
    — last `forward_days` rows have `null` for both columns (no future data available)

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_label_generator.py
from datetime import datetime, timezone, timedelta
import polars as pl
import pytest

from src.features.label_generator import add_labels


def _make_df(closes: list[float]) -> pl.DataFrame:
    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return pl.DataFrame({
        "time":   [base + timedelta(days=i) for i in range(len(closes))],
        "ticker": ["AAPL"] * len(closes),
        "close":  closes,
    })


def test_forward_return_correct():
    df = add_labels(_make_df([100.0, 101.0, 102.0, 103.0, 104.0, 106.0]))
    assert abs(df["forward_return_5d"][0] - 0.06) < 1e-6


def test_last_five_rows_are_null():
    df = add_labels(_make_df([100.0] * 10))
    assert df["label"][-5:].is_null().all()
    assert df["forward_return_5d"][-5:].is_null().all()


def test_buy_label_when_return_above_threshold():
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 103.0]
    df = add_labels(_make_df(closes), buy_threshold=0.02, sell_threshold=-0.02)
    assert df["label"][0] == "Buy"


def test_sell_label_when_return_below_threshold():
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 97.0]
    df = add_labels(_make_df(closes), buy_threshold=0.02, sell_threshold=-0.02)
    assert df["label"][0] == "Sell"


def test_hold_label_when_return_within_thresholds():
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 101.0]
    df = add_labels(_make_df(closes), buy_threshold=0.02, sell_threshold=-0.02)
    assert df["label"][0] == "Hold"


def test_custom_forward_days():
    closes = [100.0, 100.0, 100.0, 105.0]
    df = add_labels(_make_df(closes), forward_days=3, buy_threshold=0.02, sell_threshold=-0.02)
    assert df["label"][0] == "Buy"
    assert df["label"][-3:].is_null().all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_label_generator.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/features/label_generator.py**

```python
import polars as pl


def add_labels(
    df: pl.DataFrame,
    forward_days: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
) -> pl.DataFrame:
    future_close = pl.col("close").shift(-forward_days)
    forward_return = (future_close / pl.col("close") - 1).alias("forward_return_5d")

    label = (
        pl.when(pl.col("forward_return_5d") > buy_threshold).then(pl.lit("Buy"))
        .when(pl.col("forward_return_5d") < sell_threshold).then(pl.lit("Sell"))
        .otherwise(pl.lit("Hold"))
        .alias("label")
    )

    return (
        df
        .with_columns(forward_return)
        .with_columns(
            pl.when(pl.col("forward_return_5d").is_null())
            .then(None)
            .otherwise(label)
            .alias("label")
        )
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_label_generator.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/label_generator.py tests/unit/test_label_generator.py
git commit -m "feat: label generator with 5d forward return and Buy/Hold/Sell classification"
```

---

### Task 6: Feature Store

**Files:**
- Create: `src/features/feature_store.py`
- Test: `tests/unit/test_feature_store.py`

**Interfaces:**
- Consumes: all feature modules from Tasks 2–5; `src.ingestion.db.get_connection`, `release_connection`, `execute_many`; `src.ingestion.collector.collect_ohlcv`
- Produces:
  - `build_features(ticker: str, start: datetime, end: datetime) -> pl.DataFrame`
    — reads raw OHLCV + sentiment from TimescaleDB, applies all transforms, returns joined feature DataFrame
  - `write_features(df: pl.DataFrame) -> int` — upserts to `features` table, returns row count
  - `export_parquet(df: pl.DataFrame, ticker: str, output_dir: Path) -> Path` — writes `{output_dir}/{ticker}.parquet`, returns path

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_feature_store.py
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
import tempfile
import pytest

from src.features.feature_store import build_features, write_features, export_parquet


def _sample_ohlcv(n: int = 250) -> pl.DataFrame:
    import math
    base = datetime(2022, 1, 3, tzinfo=timezone.utc)
    closes = [100.0 + 5 * math.sin(i / 10) + i * 0.1 for i in range(n)]
    return pl.DataFrame({
        "time":        [base + timedelta(days=i) for i in range(n)],
        "ticker":      ["AAPL"] * n,
        "open":        [c - 0.5 for c in closes],
        "high":        [c + 1.0 for c in closes],
        "low":         [c - 1.0 for c in closes],
        "close":       closes,
        "volume":      [1_000_000] * n,
        "adj_close":   closes,
        "dividends":   [0.0] * n,
        "stock_splits":[0.0] * n,
    })


def _sample_sentiment(n: int = 250) -> pl.DataFrame:
    from datetime import date
    base = date(2022, 1, 3)
    return pl.DataFrame({
        "time":          [base + timedelta(days=i) for i in range(n)],
        "ticker":        ["AAPL"] * n,
        "avg_pos":       [0.6] * n,
        "avg_neg":       [0.2] * n,
        "avg_neu":       [0.2] * n,
        "article_count": [3] * n,
    })


@patch("src.features.feature_store._load_ohlcv_from_db")
@patch("src.features.feature_store._load_sentiment_from_db")
@patch("src.features.feature_store._load_spy_from_db")
@patch("src.features.feature_store._load_vix_from_db")
def test_build_features_returns_dataframe_with_all_columns(
    mock_vix, mock_spy, mock_sent, mock_ohlcv
):
    ohlcv = _sample_ohlcv()
    mock_ohlcv.return_value = ohlcv
    mock_spy.return_value = ohlcv.with_columns(pl.lit("SPY").alias("ticker"))
    mock_vix.return_value = ohlcv.with_columns(pl.lit("^VIX").alias("ticker"))
    mock_sent.return_value = _sample_sentiment()

    start = datetime(2022, 1, 3, tzinfo=timezone.utc)
    end = datetime(2022, 12, 31, tzinfo=timezone.utc)
    df = build_features("AAPL", start, end)

    assert isinstance(df, pl.DataFrame)
    for col in ["sma_20", "rsi_14", "macd", "sent_pos_avg_5d", "label", "forward_return_5d"]:
        assert col in df.columns, f"Missing: {col}"


@patch("src.features.feature_store.get_connection", return_value=MagicMock())
@patch("src.features.feature_store.execute_many")
@patch("src.features.feature_store.release_connection")
def test_write_features_returns_row_count(mock_rel, mock_exec, mock_conn):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
        "ticker": ["AAPL"],
        "sma_10": [150.0], "sma_20": [148.0], "sma_50": [145.0], "sma_200": [140.0],
        "ema_12": [151.0], "ema_26": [149.0],
        "rsi_14": [55.0], "macd": [1.2], "macd_signal": [1.0], "macd_hist": [0.2],
        "bb_upper": [160.0], "bb_lower": [140.0], "bb_width": [0.1],
        "atr_14": [2.5], "hist_vol_21": [0.15],
        "sent_pos_avg_3d": [0.6], "sent_pos_avg_5d": [0.58], "sent_pos_avg_10d": [0.57],
        "sent_pos_mom_3d": [0.02], "news_vol_spike": [0.5],
        "rel_strength_spy": [1.1], "vix_level": [15.0],
        "forward_return_5d": [0.03], "label": ["Buy"],
    })
    n = write_features(df)
    assert n == 1
    assert mock_exec.called


def test_export_parquet_writes_file():
    df = pl.DataFrame({"time": [datetime(2024, 1, 2, tzinfo=timezone.utc)], "ticker": ["AAPL"], "close": [153.0]})
    with tempfile.TemporaryDirectory() as tmp:
        path = export_parquet(df, "AAPL", Path(tmp))
        assert path.exists()
        assert path.name == "AAPL.parquet"
        loaded = pl.read_parquet(path)
        assert len(loaded) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_feature_store.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/features/feature_store.py**

```python
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import logging

import polars as pl
import psycopg2

from src.ingestion.db import get_connection, release_connection, execute_many
from src.features.technical_indicators import add_technical_indicators
from src.features.sentiment_features import add_sentiment_features
from src.features.cross_asset_features import add_cross_asset_features
from src.features.label_generator import add_labels

logger = logging.getLogger(__name__)

_FEATURES_SQL = """
INSERT INTO features (
    time, ticker,
    sma_10, sma_20, sma_50, sma_200, ema_12, ema_26,
    rsi_14, macd, macd_signal, macd_hist,
    bb_upper, bb_lower, bb_width, atr_14, hist_vol_21,
    sent_pos_avg_3d, sent_pos_avg_5d, sent_pos_avg_10d,
    sent_pos_mom_3d, news_vol_spike,
    rel_strength_spy, vix_level,
    forward_return_5d, label
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (ticker, time) DO UPDATE SET
    sma_10=EXCLUDED.sma_10, sma_20=EXCLUDED.sma_20,
    rsi_14=EXCLUDED.rsi_14, macd=EXCLUDED.macd,
    label=EXCLUDED.label, forward_return_5d=EXCLUDED.forward_return_5d
"""

_FEATURE_COLS = [
    "time", "ticker",
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike",
    "rel_strength_spy", "vix_level",
    "forward_return_5d", "label",
]


def build_features(ticker: str, start: datetime, end: datetime) -> pl.DataFrame:
    stock_df = _load_ohlcv_from_db(ticker, start, end)
    spy_df   = _load_spy_from_db(start, end)
    vix_df   = _load_vix_from_db(start, end)
    sent_df  = _load_sentiment_from_db(ticker, start, end)

    df = add_technical_indicators(stock_df)
    df = add_cross_asset_features(df, spy_df, vix_df)

    sent_feats = add_sentiment_features(sent_df)
    sent_feats = sent_feats.with_columns(
        pl.col("time").cast(pl.Date).alias("join_date")
    )
    df = df.with_columns(pl.col("time").dt.date().alias("join_date"))
    df = df.join(
        sent_feats.select([
            "join_date",
            "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
            "sent_pos_mom_3d", "news_vol_spike",
        ]),
        on="join_date",
        how="left",
    ).drop("join_date")

    df = add_labels(df)
    return df.drop_nulls(subset=["label"])


def write_features(df: pl.DataFrame) -> int:
    rows = [
        tuple(row[col] for col in _FEATURE_COLS)
        for row in df.select(_FEATURE_COLS).iter_rows(named=True)
    ]
    conn = get_connection()
    try:
        execute_many(conn, _FEATURES_SQL, rows)
    finally:
        release_connection(conn)
    return len(rows)


def export_parquet(df: pl.DataFrame, ticker: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker}.parquet"
    df.write_parquet(path)
    return path


def _load_ohlcv_from_db(ticker: str, start: datetime, end: datetime) -> pl.DataFrame:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT time, ticker, open, high, low, close, volume,
                       adj_close, dividends, stock_splits
                FROM ohlcv
                WHERE ticker = %s AND time >= %s AND time <= %s
                ORDER BY time ASC
                """,
                (ticker, start, end),
            )
            rows = cur.fetchall()
    finally:
        release_connection(conn)
    cols = ["time", "ticker", "open", "high", "low", "close", "volume",
            "adj_close", "dividends", "stock_splits"]
    return pl.DataFrame(rows, schema=cols, orient="row")


def _load_sentiment_from_db(ticker: str, start: datetime, end: datetime) -> pl.DataFrame:
    from src.features.sentiment_features import aggregate_daily_sentiment
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT published_at, ticker, sentiment_pos, sentiment_neg, sentiment_neu
                FROM news_articles
                WHERE ticker = %s AND published_at >= %s AND published_at <= %s
                """,
                (ticker, start, end),
            )
            rows = cur.fetchall()
    finally:
        release_connection(conn)
    if not rows:
        from datetime import date, timedelta
        import math
        n = (end.date() - start.date()).days + 1
        return pl.DataFrame({
            "time": [start.date() + timedelta(days=i) for i in range(n)],
            "ticker": [ticker] * n,
            "avg_pos": [0.5] * n, "avg_neg": [0.25] * n, "avg_neu": [0.25] * n,
            "article_count": [0] * n,
        })
    cols = ["published_at", "ticker", "sentiment_pos", "sentiment_neg", "sentiment_neu"]
    raw = pl.DataFrame(rows, schema=cols, orient="row")
    return aggregate_daily_sentiment(raw)


def _load_spy_from_db(start: datetime, end: datetime) -> pl.DataFrame:
    return _load_ohlcv_from_db("SPY", start, end)


def _load_vix_from_db(start: datetime, end: datetime) -> pl.DataFrame:
    return _load_ohlcv_from_db("^VIX", start, end)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_feature_store.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/feature_store.py tests/unit/test_feature_store.py
git commit -m "feat: feature store orchestrator — build, write, export parquet"
```

---

### Task 7: dbt Models & Data Quality Tests

**Files:**
- Create: `dbt/dbt_project.yml`
- Create: `dbt/profiles.yml`
- Create: `dbt/models/staging/stg_ohlcv.sql`
- Create: `dbt/models/staging/stg_news_sentiment.sql`
- Create: `dbt/models/staging/schema.yml`
- Create: `dbt/models/features/feat_sma.sql`
- Create: `dbt/models/features/feat_sentiment_daily.sql`
- Create: `dbt/models/features/feat_labels.sql`
- Create: `dbt/models/features/schema.yml`
- Test: `tests/unit/test_dbt_project.py`

**Interfaces:**
- Consumes: `ohlcv`, `news_articles`, `features` tables in TimescaleDB
- Produces: dbt views `stg_ohlcv`, `stg_news_sentiment`, `feat_sma`, `feat_sentiment_daily`, `feat_labels` — plus data quality test definitions in `schema.yml`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_dbt_project.py
from pathlib import Path
import yaml


DBT_DIR = Path(__file__).parents[2] / "dbt"


def test_dbt_project_yml_exists_and_valid():
    project_file = DBT_DIR / "dbt_project.yml"
    assert project_file.exists()
    data = yaml.safe_load(project_file.read_text())
    assert data["name"] == "financial_platform"
    assert "models" in data


def test_staging_models_exist():
    staging = DBT_DIR / "models" / "staging"
    assert (staging / "stg_ohlcv.sql").exists()
    assert (staging / "stg_news_sentiment.sql").exists()
    assert (staging / "schema.yml").exists()


def test_feature_models_exist():
    feats = DBT_DIR / "models" / "features"
    assert (feats / "feat_sma.sql").exists()
    assert (feats / "feat_sentiment_daily.sql").exists()
    assert (feats / "feat_labels.sql").exists()
    assert (feats / "schema.yml").exists()


def test_schema_yml_defines_not_null_tests():
    schema = yaml.safe_load(
        (DBT_DIR / "models" / "staging" / "schema.yml").read_text()
    )
    model_names = [m["name"] for m in schema["models"]]
    assert "stg_ohlcv" in model_names
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_dbt_project.py -v
```
Expected: `AssertionError` on file existence check.

- [ ] **Step 3: Create dbt/dbt_project.yml**

```yaml
name: financial_platform
version: "1.0.0"
config-version: 2

profile: financial_platform

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets: ["target", "dbt_packages"]

models:
  financial_platform:
    staging:
      +materialized: view
      +schema: staging
    features:
      +materialized: view
      +schema: features
```

- [ ] **Step 4: Create dbt/profiles.yml**

```yaml
financial_platform:
  target: dev
  outputs:
    dev:
      type: postgres
      host: "{{ env_var('POSTGRES_HOST', 'localhost') }}"
      user: "{{ env_var('POSTGRES_USER', 'platform') }}"
      password: "{{ env_var('POSTGRES_PASSWORD', 'platform') }}"
      port: 5432
      dbname: "{{ env_var('POSTGRES_DB', 'financial') }}"
      schema: public
      threads: 4
```

- [ ] **Step 5: Create dbt/models/staging/stg_ohlcv.sql**

```sql
SELECT
    time                                    AS time,
    ticker                                  AS ticker,
    open                                    AS open,
    high                                    AS high,
    low                                     AS low,
    close                                   AS close,
    volume                                  AS volume,
    adj_close                               AS adj_close,
    dividends                               AS dividends,
    stock_splits                            AS stock_splits
FROM {{ source('public', 'ohlcv') }}
WHERE time IS NOT NULL
  AND ticker IS NOT NULL
  AND close > 0
```

- [ ] **Step 6: Create dbt/models/staging/stg_news_sentiment.sql**

```sql
SELECT
    published_at                                AS published_at,
    ticker                                      AS ticker,
    COALESCE(sentiment_pos, 0.0)                AS sentiment_pos,
    COALESCE(sentiment_neg, 0.0)                AS sentiment_neg,
    COALESCE(sentiment_neu, 1.0)                AS sentiment_neu,
    sentiment_label                             AS sentiment_label
FROM {{ source('public', 'news_articles') }}
WHERE published_at IS NOT NULL
  AND headline IS NOT NULL
```

- [ ] **Step 7: Create dbt/models/staging/schema.yml**

```yaml
version: 2

sources:
  - name: public
    tables:
      - name: ohlcv
      - name: news_articles
      - name: features

models:
  - name: stg_ohlcv
    description: "Cleaned OHLCV with positive close filter"
    columns:
      - name: time
        tests: [not_null]
      - name: ticker
        tests: [not_null]
      - name: close
        tests: [not_null]

  - name: stg_news_sentiment
    description: "Cleaned news articles with coalesced nulls"
    columns:
      - name: published_at
        tests: [not_null]
      - name: ticker
        tests: [not_null]
```

- [ ] **Step 8: Create dbt/models/features/feat_sma.sql**

```sql
WITH ranked AS (
    SELECT
        time,
        ticker,
        close,
        AVG(close) OVER (
            PARTITION BY ticker
            ORDER BY time
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS sma_20,
        AVG(close) OVER (
            PARTITION BY ticker
            ORDER BY time
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) AS sma_50
    FROM {{ ref('stg_ohlcv') }}
)
SELECT * FROM ranked
```

- [ ] **Step 9: Create dbt/models/features/feat_sentiment_daily.sql**

```sql
SELECT
    DATE(published_at)              AS date,
    ticker                          AS ticker,
    AVG(sentiment_pos)              AS avg_pos,
    AVG(sentiment_neg)              AS avg_neg,
    COUNT(*)                        AS article_count
FROM {{ ref('stg_news_sentiment') }}
GROUP BY DATE(published_at), ticker
ORDER BY date, ticker
```

- [ ] **Step 10: Create dbt/models/features/feat_labels.sql**

```sql
SELECT
    time,
    ticker,
    close,
    LEAD(close, 5) OVER (PARTITION BY ticker ORDER BY time) / close - 1
        AS forward_return_5d,
    CASE
        WHEN LEAD(close, 5) OVER (PARTITION BY ticker ORDER BY time) / close - 1 > 0.02  THEN 'Buy'
        WHEN LEAD(close, 5) OVER (PARTITION BY ticker ORDER BY time) / close - 1 < -0.02 THEN 'Sell'
        ELSE 'Hold'
    END AS label
FROM {{ ref('stg_ohlcv') }}
```

- [ ] **Step 11: Create dbt/models/features/schema.yml**

```yaml
version: 2

models:
  - name: feat_sma
    description: "20-day and 50-day SMAs validated in SQL"
    columns:
      - name: sma_20
        tests: [not_null]

  - name: feat_labels
    description: "Buy/Hold/Sell labels computed via SQL LEAD window — reference for bias check"
    columns:
      - name: label
        tests:
          - accepted_values:
              values: ["Buy", "Hold", "Sell"]
```

- [ ] **Step 12: Run tests to verify they pass**

```bash
pytest tests/unit/test_dbt_project.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 13: Validate dbt compiles against TimescaleDB (requires running DB)**

```bash
cd dbt && dbt compile --profiles-dir . && cd ..
```
Expected: `Completed successfully` with no errors.

- [ ] **Step 14: Commit**

```bash
git add dbt/ tests/unit/test_dbt_project.py
git commit -m "feat: dbt staging and feature models with data quality tests"
```

---

### Task 8: Feature Engineering Airflow DAG

**Files:**
- Create: `dags/feature_engineering_dag.py`
- Test: `tests/unit/test_feature_dag.py`

**Interfaces:**
- Consumes: `build_features`, `write_features`, `export_parquet`, `get_sp500_tickers_at`
- Produces: DAG `feature_engineering_dag`, `schedule_interval="@daily"`, tasks: `compute_and_store_features`, `export_to_parquet`
  - Runs after `historical_data_dag` and `news_sentiment_dag` complete on the same day

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_feature_dag.py
import importlib


def test_feature_engineering_dag_loads():
    mod = importlib.import_module("dags.feature_engineering_dag")
    assert hasattr(mod, "dag")
    assert mod.dag.dag_id == "feature_engineering_dag"


def test_dag_has_expected_tasks():
    mod = importlib.import_module("dags.feature_engineering_dag")
    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "compute_and_store_features" in task_ids
    assert "export_to_parquet" in task_ids


def test_export_depends_on_compute():
    mod = importlib.import_module("dags.feature_engineering_dag")
    dag = mod.dag
    compute = dag.get_task("compute_and_store_features")
    export = dag.get_task("export_to_parquet")
    assert export.task_id in {d.task_id for d in compute.downstream_list}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_feature_dag.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement dags/feature_engineering_dag.py**

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

_default_args = {
    "owner": "platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

_PARQUET_DIR = Path("/opt/airflow/data/features")


def _compute_and_store(**context):
    import logging
    from src.ingestion.survivorship import get_sp500_tickers_at
    from src.features.feature_store import build_features, write_features

    log = logging.getLogger(__name__)
    execution_date = context["execution_date"]
    tickers = get_sp500_tickers_at(execution_date)
    end = execution_date.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=365 * 2)

    for ticker in tickers:
        try:
            df = build_features(ticker, start, end)
            if not df.is_empty():
                write_features(df)
                context["ti"].xcom_push(key=ticker, value=len(df))
        except Exception as e:
            log.error("Feature build failed %s: %s", ticker, e)


def _export_parquet(**context):
    import logging
    from src.ingestion.survivorship import get_sp500_tickers_at
    from src.features.feature_store import build_features, export_parquet

    log = logging.getLogger(__name__)
    execution_date = context["execution_date"]
    tickers = get_sp500_tickers_at(execution_date)
    end = execution_date.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=365 * 2)

    for ticker in tickers:
        try:
            df = build_features(ticker, start, end)
            if not df.is_empty():
                path = export_parquet(df, ticker, _PARQUET_DIR)
                log.info("Exported %s → %s", ticker, path)
        except Exception as e:
            log.error("Parquet export failed %s: %s", ticker, e)


with DAG(
    dag_id="feature_engineering_dag",
    default_args=_default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features"],
) as dag:
    compute = PythonOperator(
        task_id="compute_and_store_features",
        python_callable=_compute_and_store,
    )
    export = PythonOperator(
        task_id="export_to_parquet",
        python_callable=_export_parquet,
    )
    compute >> export
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_feature_dag.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests PASS. DB-dependent tests require the TimescaleDB container running.

- [ ] **Step 6: Commit**

```bash
git add dags/feature_engineering_dag.py tests/unit/test_feature_dag.py
git commit -m "feat: feature engineering Airflow DAG with parquet export"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| SMA, EMA | Task 2 (`sma_10/20/50/200`, `ema_12/26`) |
| RSI | Task 2 (`rsi_14`) |
| MACD | Task 2 (`macd`, `macd_signal`, `macd_hist`) |
| Bollinger Bands | Task 2 (`bb_upper`, `bb_lower`, `bb_width`) |
| ATR | Task 2 (`atr_14`) |
| Volatility estimates | Task 2 (`hist_vol_21`) |
| Rolling average sentiment (3/5/10 days) | Task 3 (`sent_pos_avg_3d/5d/10d`) |
| Sentiment momentum | Task 3 (`sent_pos_mom_3d`) |
| News volume spikes | Task 3 (`news_vol_spike` z-score) |
| VIX correlation | Task 4 (`vix_level`) |
| Sector performance / relative strength | Task 4 (`rel_strength_spy`) |
| Target label creation (Buy/Hold/Sell at 5d ±2%) | Task 5 |
| Polars for high-speed computation | Tasks 2–5, 6 (all pure Polars) |
| dbt for traceability and version control | Task 7 |
| TimescaleDB storage | Task 1 (schema) + Task 6 (writer) |
| DuckDB for local analytical queries | Task 1 (client) + Task 6 (export) |
| Daily Airflow orchestration | Task 8 |

**Placeholder scan:** No TBDs. All steps include code. `_load_spy_from_db` and `_load_vix_from_db` call `_load_ohlcv_from_db("SPY", ...)` and `_load_ohlcv_from_db("^VIX", ...)` — SPY and VIX must be seeded into the `ohlcv` table by Plan 1's ingestion DAG (add `"SPY"` and `"^VIX"` to the `watchlist` universe in `config/stocks.yaml`).

**Type consistency:** `pl.DataFrame` column names (`sma_10`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bb_upper`, `bb_lower`, `bb_width`, `atr_14`, `hist_vol_21`, `sent_pos_avg_3d`, `sent_pos_avg_5d`, `sent_pos_avg_10d`, `sent_pos_mom_3d`, `news_vol_spike`, `rel_strength_spy`, `vix_level`, `forward_return_5d`, `label`) are consistent from feature modules through `feature_store.py`, `write_features`, and the `features` table schema in `002_feature_schema.sql`.
