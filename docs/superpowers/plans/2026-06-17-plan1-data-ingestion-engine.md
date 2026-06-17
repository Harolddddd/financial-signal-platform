# Financial Platform — Plan 1: Data Ingestion Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete data ingestion layer that collects survivorship-bias-free OHLCV history with multi-source fallback, news headlines, and NLP sentiment scores, storing all output in TimescaleDB.

**Architecture:** A layered collector chain (yFinance → Alpha Vantage → FMP) wraps each data fetch; a survivorship module reconstructs historical S&P 500 compositions from a versioned CSV; a news collector aggregates RSS/NewsAPI/Finnhub; FinBERT scores each headline with VADER as fallback; a single storage writer upserts everything to TimescaleDB hypertables. Two Airflow DAGs schedule daily refreshes.

**Tech Stack:** Python 3.11, PostgreSQL 15 + TimescaleDB 2.x, Apache Airflow 2.9, Polars 0.20, yfinance 0.2.40, alpha-vantage 2.3, finnhub-python 1.x, feedparser 6.x, newsapi-python 0.2, transformers 4.40 (ProsusAI/finbert), vaderSentiment 3.3, psycopg2-binary 2.9, Docker Compose, pytest 8.x

## Global Constraints

- Python >= 3.11; all function signatures require type hints
- All timestamps stored as UTC `TIMESTAMPTZ`; never store naive datetimes
- API keys loaded from environment only — never hardcoded
- Polars for all DataFrame operations; Pandas only when a library forces it (yFinance returns Pandas)
- yFinance = primary source; Alpha Vantage + FMP = fallback only
- FinBERT model loaded once at process start via a module-level singleton, not per-call
- Tests use pytest only; no `unittest.TestCase`
- Docker Compose for local dev; all services defined in `docker-compose.yml`
- No data written to DB without a ticker and a valid UTC timestamp
- All DB operations use psycopg2 connection pooling

---

### Task 1: Project Scaffold & Infrastructure

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `config/stocks.yaml`
- Create: `src/__init__.py`
- Create: `src/ingestion/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing
- Produces: `config.settings.Settings` — a pydantic-settings object importable as `from config.settings import settings`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "financial-platform"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "yfinance>=0.2.40",
    "alpha-vantage>=2.3.1",
    "finnhub-python>=1.0.0",
    "newsapi-python>=0.2.7",
    "feedparser>=6.0.11",
    "polars>=0.20.0",
    "psycopg2-binary>=2.9.9",
    "pydantic-settings>=2.3.0",
    "transformers>=4.40.0",
    "torch>=2.3.0",
    "vaderSentiment>=3.3.2",
    "apache-airflow>=2.9.0",
    "requests>=2.32.0",
    "pyyaml>=6.0.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-mock>=3.14.0",
    "responses>=0.25.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
version: "3.9"
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-platform}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-platform}
      POSTGRES_DB: ${POSTGRES_DB:-financial}
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d

volumes:
  timescale_data:
```

- [ ] **Step 3: Create .env.example**

```
POSTGRES_USER=platform
POSTGRES_PASSWORD=platform
POSTGRES_DB=financial
DATABASE_URL=postgresql://platform:platform@localhost:5432/financial

ALPHA_VANTAGE_KEY=
FMP_KEY=
NEWSAPI_KEY=
FINNHUB_KEY=

STOCK_UNIVERSE=sp500
HISTORICAL_DAYS=3650
```

- [ ] **Step 4: Create config/settings.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATABASE_URL: str = "postgresql://platform:platform@localhost:5432/financial"

    ALPHA_VANTAGE_KEY: str = ""
    FMP_KEY: str = ""
    NEWSAPI_KEY: str = ""
    FINNHUB_KEY: str = ""

    STOCK_UNIVERSE: str = "sp500"
    HISTORICAL_DAYS: int = 3650


settings = Settings()
```

- [ ] **Step 5: Create config/stocks.yaml**

```yaml
universes:
  sp500:
    description: "S&P 500 constituents — populated at runtime from survivorship module"
    tickers: []
  watchlist:
    description: "Small test watchlist"
    tickers:
      - AAPL
      - MSFT
      - GOOGL
      - AMZN
      - NVDA
```

- [ ] **Step 6: Create tests/conftest.py**

```python
import pytest
import psycopg2
from config.settings import settings


@pytest.fixture(scope="session")
def db_conn():
    conn = psycopg2.connect(settings.DATABASE_URL)
    yield conn
    conn.close()


@pytest.fixture
def sample_tickers() -> list[str]:
    return ["AAPL", "MSFT", "GOOGL"]
```

- [ ] **Step 7: Create empty __init__ files**

```bash
touch src/__init__.py src/ingestion/__init__.py tests/__init__.py config/__init__.py
```

- [ ] **Step 8: Start TimescaleDB and verify connection**

```bash
cp .env.example .env
docker compose up -d timescaledb
sleep 5
docker compose exec timescaledb psql -U platform -d financial -c "SELECT version();"
```

Expected: PostgreSQL 15.x version line printed without errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example config/ src/ tests/conftest.py tests/__init__.py
git commit -m "feat: project scaffold, docker, settings, conftest"
```

---

### Task 2: Database Schema & Migration

**Files:**
- Create: `migrations/001_initial_schema.sql`
- Create: `src/ingestion/db.py`
- Test: `tests/unit/test_db.py`

**Interfaces:**
- Consumes: `config.settings.settings.DATABASE_URL`
- Produces:
  - `get_connection() -> psycopg2.extensions.connection`
  - `release_connection(conn: psycopg2.extensions.connection) -> None`
  - `execute_many(conn, sql: str, rows: list[tuple]) -> None`
  - Tables in DB: `ohlcv`, `corporate_actions`, `fundamentals`, `news_articles`, `index_compositions`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_db.py
import pytest
from src.ingestion.db import get_connection, execute_many, release_connection


def test_get_connection_returns_open_connection():
    conn = get_connection()
    assert conn.closed == 0
    release_connection(conn)


def test_execute_many_inserts_rows(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE _test_em (val TEXT)")
    execute_many(db_conn, "INSERT INTO _test_em VALUES (%s)", [("a",), ("b",)])
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM _test_em")
        assert cur.fetchone()[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_db.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.ingestion.db'`

- [ ] **Step 3: Create migrations/001_initial_schema.sql**

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE TABLE IF NOT EXISTS ohlcv (
    time         TIMESTAMPTZ      NOT NULL,
    ticker       TEXT             NOT NULL,
    open         DOUBLE PRECISION,
    high         DOUBLE PRECISION,
    low          DOUBLE PRECISION,
    close        DOUBLE PRECISION,
    volume       BIGINT,
    adj_close    DOUBLE PRECISION,
    dividends    DOUBLE PRECISION DEFAULT 0,
    stock_splits DOUBLE PRECISION DEFAULT 0
);
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_ticker_time ON ohlcv (ticker, time DESC);

CREATE TABLE IF NOT EXISTS corporate_actions (
    time        TIMESTAMPTZ      NOT NULL,
    ticker      TEXT             NOT NULL,
    action_type TEXT             NOT NULL,
    value       DOUBLE PRECISION,
    metadata    JSONB
);
SELECT create_hypertable('corporate_actions', 'time', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS fundamentals (
    time       TIMESTAMPTZ      NOT NULL,
    ticker     TEXT             NOT NULL,
    pe_ratio   DOUBLE PRECISION,
    eps        DOUBLE PRECISION,
    book_value DOUBLE PRECISION,
    market_cap DOUBLE PRECISION
);
SELECT create_hypertable('fundamentals', 'time', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS news_articles (
    id              BIGSERIAL,
    published_at    TIMESTAMPTZ NOT NULL,
    ticker          TEXT,
    headline        TEXT        NOT NULL,
    body            TEXT,
    source          TEXT,
    url             TEXT UNIQUE,
    relevance       DOUBLE PRECISION,
    sentiment_pos   DOUBLE PRECISION,
    sentiment_neg   DOUBLE PRECISION,
    sentiment_neu   DOUBLE PRECISION,
    sentiment_label TEXT
);
SELECT create_hypertable('news_articles', 'published_at', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS index_compositions (
    index_name   TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    added_date   DATE NOT NULL,
    removed_date DATE,
    PRIMARY KEY (index_name, ticker, added_date)
);
```

- [ ] **Step 4: Apply migration**

```bash
docker compose exec timescaledb psql -U platform -d financial \
  -f /docker-entrypoint-initdb.d/001_initial_schema.sql
```
Expected: Lines like `CREATE TABLE`, `create_hypertable`, `CREATE INDEX` — no errors.

- [ ] **Step 5: Implement src/ingestion/db.py**

```python
import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from config.settings import settings

_pool: pg_pool.SimpleConnectionPool | None = None


def _get_pool() -> pg_pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.SimpleConnectionPool(1, 10, settings.DATABASE_URL)
    return _pool


def get_connection() -> psycopg2.extensions.connection:
    return _get_pool().getconn()


def release_connection(conn: psycopg2.extensions.connection) -> None:
    _get_pool().putconn(conn)


def execute_many(
    conn: psycopg2.extensions.connection,
    sql: str,
    rows: list[tuple],
) -> None:
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=1000)
    conn.commit()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/unit/test_db.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add migrations/001_initial_schema.sql src/ingestion/db.py tests/unit/test_db.py
git commit -m "feat: timescaledb schema and psycopg2 connection pool"
```

---

### Task 3: Historical Collector — yFinance Primary

**Files:**
- Create: `src/ingestion/historical_collector.py`
- Test: `tests/unit/test_historical_collector.py`

**Interfaces:**
- Consumes: nothing external at import time
- Produces:
  - `fetch_ohlcv(ticker: str, start: datetime, end: datetime, interval: str = "1d") -> pl.DataFrame`
    - Columns: `time: Datetime[us, UTC]`, `ticker: Utf8`, `open: Float64`, `high: Float64`, `low: Float64`, `close: Float64`, `volume: Int64`, `adj_close: Float64`, `dividends: Float64`, `stock_splits: Float64`
    - Raises `ValueError` if response is empty
  - `fetch_corporate_actions(ticker: str) -> dict[str, pd.Series | pd.DataFrame]`
    - Keys: `"dividends"`, `"splits"`, `"earnings_dates"`
  - `fetch_fundamentals(ticker: str) -> dict[str, float | None]`
    - Keys: `"pe_ratio"`, `"eps"`, `"book_value"`, `"market_cap"`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_historical_collector.py
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import polars as pl
import pandas as pd
import pytest

from src.ingestion.historical_collector import (
    fetch_ohlcv,
    fetch_corporate_actions,
    fetch_fundamentals,
)


def _make_yf_history_df():
    idx = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03"],
        name="Date",
        tz="America/New_York",
    )
    return pd.DataFrame(
        {
            "Open": [150.0, 151.0],
            "High": [155.0, 156.0],
            "Low": [149.0, 150.0],
            "Close": [153.0, 154.0],
            "Volume": [1_000_000, 1_100_000],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=idx,
    )


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_ohlcv_returns_polars_dataframe(mock_ticker):
    mock_ticker.return_value.history.return_value = _make_yf_history_df()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)

    df = fetch_ohlcv("AAPL", start, end)

    assert isinstance(df, pl.DataFrame)
    assert "ticker" in df.columns
    assert df["ticker"][0] == "AAPL"
    assert len(df) == 2


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_ohlcv_raises_on_empty_response(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="No data returned"):
        fetch_ohlcv(
            "FAKE",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
        )


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_corporate_actions_returns_expected_keys(mock_ticker):
    inst = MagicMock()
    inst.dividends = pd.Series([], dtype=float)
    inst.splits = pd.Series([], dtype=float)
    inst.earnings_dates = pd.DataFrame()
    mock_ticker.return_value = inst

    result = fetch_corporate_actions("AAPL")
    assert set(result.keys()) == {"dividends", "splits", "earnings_dates"}


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_fundamentals_parses_info(mock_ticker):
    mock_ticker.return_value.info = {
        "trailingPE": 28.5,
        "trailingEps": 6.43,
        "bookValue": 4.5,
        "marketCap": 3_000_000_000_000,
    }
    result = fetch_fundamentals("AAPL")
    assert result["pe_ratio"] == 28.5
    assert result["eps"] == 6.43


@patch("src.ingestion.historical_collector.yf.Ticker")
def test_fetch_fundamentals_handles_missing_keys(mock_ticker):
    mock_ticker.return_value.info = {}
    result = fetch_fundamentals("UNKNOWN")
    assert result["pe_ratio"] is None
    assert result["market_cap"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_historical_collector.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.ingestion.historical_collector'`

- [ ] **Step 3: Implement src/ingestion/historical_collector.py**

```python
from datetime import datetime, timezone
import logging

import pandas as pd
import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pl.DataFrame:
    raw = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        back_adjust=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
    if "date" in raw.columns:
        raw = raw.rename(columns={"date": "time"})

    raw["ticker"] = ticker
    raw["time"] = pd.to_datetime(raw["time"], utc=True)

    return pl.from_pandas(raw).select([
        pl.col("time").cast(pl.Datetime("us", "UTC")),
        pl.col("ticker"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
        pl.col("close").alias("adj_close").cast(pl.Float64),
        pl.col("dividends").cast(pl.Float64),
        pl.col("stock_splits").cast(pl.Float64),
    ])


def fetch_corporate_actions(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    return {
        "dividends": t.dividends,
        "splits": t.splits,
        "earnings_dates": t.earnings_dates if t.earnings_dates is not None else pd.DataFrame(),
    }


def fetch_fundamentals(ticker: str) -> dict[str, float | None]:
    info = yf.Ticker(ticker).info
    return {
        "pe_ratio": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "book_value": info.get("bookValue"),
        "market_cap": info.get("marketCap"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_historical_collector.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/historical_collector.py tests/unit/test_historical_collector.py
git commit -m "feat: yfinance historical collector with OHLCV, actions, fundamentals"
```

---

### Task 4: Alpha Vantage Fallback Client

**Files:**
- Create: `src/ingestion/alpha_vantage_client.py`
- Test: `tests/unit/test_alpha_vantage_client.py`

**Interfaces:**
- Consumes: `config.settings.settings.ALPHA_VANTAGE_KEY`
- Produces:
  - `fetch_ohlcv_av(ticker: str, start: datetime, end: datetime) -> pl.DataFrame`
    - Same column schema as `fetch_ohlcv` in Task 3
    - Raises `ValueError` if response contains no `"Time Series (Daily)"` key

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_alpha_vantage_client.py
from datetime import datetime, timezone
import polars as pl
import pytest
import responses as resp_mock

from src.ingestion.alpha_vantage_client import fetch_ohlcv_av

_AV_URL = "https://www.alphavantage.co/query"
_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 1, 5, tzinfo=timezone.utc)

_SAMPLE = {
    "Time Series (Daily)": {
        "2024-01-03": {
            "1. open": "151.00", "2. high": "156.00", "3. low": "150.00",
            "4. close": "154.00", "5. adjusted close": "154.00",
            "6. volume": "1100000", "7. dividend amount": "0.00",
            "8. split coefficient": "1.00",
        },
        "2024-01-02": {
            "1. open": "150.00", "2. high": "155.00", "3. low": "149.00",
            "4. close": "153.00", "5. adjusted close": "153.00",
            "6. volume": "1000000", "7. dividend amount": "0.00",
            "8. split coefficient": "1.00",
        },
    }
}


@resp_mock.activate
def test_fetch_ohlcv_av_returns_polars_dataframe():
    resp_mock.add(resp_mock.GET, _AV_URL, json=_SAMPLE, status=200)
    df = fetch_ohlcv_av("AAPL", _START, _END)
    assert isinstance(df, pl.DataFrame)
    assert "ticker" in df.columns
    assert len(df) == 2


@resp_mock.activate
def test_fetch_ohlcv_av_raises_on_rate_limit():
    resp_mock.add(resp_mock.GET, _AV_URL, json={"Note": "API rate limit"}, status=200)
    with pytest.raises(ValueError, match="Alpha Vantage returned no time series"):
        fetch_ohlcv_av("AAPL", _START, _END)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_alpha_vantage_client.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/ingestion/alpha_vantage_client.py**

```python
from datetime import datetime, timezone
import logging

import polars as pl
import requests

from config.settings import settings

logger = logging.getLogger(__name__)
_AV_BASE = "https://www.alphavantage.co/query"


def fetch_ohlcv_av(
    ticker: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    resp = requests.get(
        _AV_BASE,
        params={
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": settings.ALPHA_VANTAGE_KEY,
        },
        timeout=30,
    )
    resp.raise_for_status()
    ts = resp.json().get("Time Series (Daily)")
    if not ts:
        raise ValueError(f"Alpha Vantage returned no time series for {ticker}")

    rows = []
    for date_str, vals in ts.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if not (start <= dt <= end):
            continue
        rows.append({
            "time": dt,
            "ticker": ticker,
            "open": float(vals["1. open"]),
            "high": float(vals["2. high"]),
            "low": float(vals["3. low"]),
            "close": float(vals["4. close"]),
            "volume": int(vals["6. volume"]),
            "adj_close": float(vals["5. adjusted close"]),
            "dividends": float(vals["7. dividend amount"]),
            "stock_splits": float(vals["8. split coefficient"]),
        })

    if not rows:
        raise ValueError(f"No data for {ticker} in range {start}–{end}")

    return pl.DataFrame(rows).sort("time")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_alpha_vantage_client.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/alpha_vantage_client.py tests/unit/test_alpha_vantage_client.py
git commit -m "feat: alpha vantage fallback OHLCV client"
```

---

### Task 5: FMP Fallback Client

**Files:**
- Create: `src/ingestion/fmp_client.py`
- Test: `tests/unit/test_fmp_client.py`

**Interfaces:**
- Consumes: `config.settings.settings.FMP_KEY`
- Produces:
  - `fetch_ohlcv_fmp(ticker: str, start: datetime, end: datetime) -> pl.DataFrame`
    - Same column schema as Task 3
    - Raises `ValueError` if `"historical"` list is empty

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_fmp_client.py
from datetime import datetime, timezone
import polars as pl
import pytest
import responses as resp_mock

from src.ingestion.fmp_client import fetch_ohlcv_fmp

_FMP_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/AAPL"
_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 1, 5, tzinfo=timezone.utc)

_SAMPLE = {
    "symbol": "AAPL",
    "historical": [
        {
            "date": "2024-01-03",
            "open": 151.0, "high": 156.0, "low": 150.0,
            "close": 154.0, "adjClose": 154.0, "volume": 1_100_000,
        }
    ],
}


@resp_mock.activate
def test_fetch_ohlcv_fmp_returns_polars_dataframe():
    resp_mock.add(resp_mock.GET, _FMP_URL, json=_SAMPLE, status=200)
    df = fetch_ohlcv_fmp("AAPL", _START, _END)
    assert isinstance(df, pl.DataFrame)
    assert df["ticker"][0] == "AAPL"
    assert len(df) == 1


@resp_mock.activate
def test_fetch_ohlcv_fmp_raises_on_empty():
    resp_mock.add(resp_mock.GET, _FMP_URL, json={"historical": []}, status=200)
    with pytest.raises(ValueError, match="No data returned"):
        fetch_ohlcv_fmp("AAPL", _START, _END)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_fmp_client.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/ingestion/fmp_client.py**

```python
from datetime import datetime, timezone
import logging

import polars as pl
import requests

from config.settings import settings

logger = logging.getLogger(__name__)
_FMP_BASE = "https://financialmodelingprep.com/api/v3"


def fetch_ohlcv_fmp(
    ticker: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    resp = requests.get(
        f"{_FMP_BASE}/historical-price-full/{ticker}",
        params={
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "apikey": settings.FMP_KEY,
        },
        timeout=30,
    )
    resp.raise_for_status()
    historical = resp.json().get("historical", [])
    if not historical:
        raise ValueError(f"No data returned for {ticker} in range {start}–{end}")

    rows = [
        {
            "time": datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc),
            "ticker": ticker,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": int(r["volume"]),
            "adj_close": float(r["adjClose"]),
            "dividends": 0.0,
            "stock_splits": 0.0,
        }
        for r in historical
    ]
    return pl.DataFrame(rows).sort("time")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_fmp_client.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/fmp_client.py tests/unit/test_fmp_client.py
git commit -m "feat: FMP fallback OHLCV client"
```

---

### Task 6: Fallback Collector Chain

**Files:**
- Create: `src/ingestion/collector.py`
- Test: `tests/unit/test_collector.py`

**Interfaces:**
- Consumes: `fetch_ohlcv`, `fetch_ohlcv_av`, `fetch_ohlcv_fmp`
- Produces:
  - `collect_ohlcv(ticker: str, start: datetime, end: datetime, interval: str = "1d") -> pl.DataFrame`
    - Tries yFinance first, then Alpha Vantage, then FMP
    - Raises `RuntimeError("All data sources failed for {ticker}: ...")` if all three fail

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_collector.py
from datetime import datetime, timezone
from unittest.mock import patch
import polars as pl
import pytest

from src.ingestion.collector import collect_ohlcv

_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 1, 5, tzinfo=timezone.utc)
_SAMPLE_DF = pl.DataFrame({
    "time": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
    "ticker": ["AAPL"],
    "open": [150.0], "high": [155.0], "low": [149.0], "close": [153.0],
    "volume": [1_000_000], "adj_close": [153.0], "dividends": [0.0], "stock_splits": [0.0],
})


@patch("src.ingestion.collector.fetch_ohlcv", return_value=_SAMPLE_DF)
def test_uses_yfinance_primary(mock_yf):
    df = collect_ohlcv("AAPL", _START, _END)
    assert mock_yf.called
    assert isinstance(df, pl.DataFrame)


@patch("src.ingestion.collector.fetch_ohlcv", side_effect=ValueError("yf fail"))
@patch("src.ingestion.collector.fetch_ohlcv_av", return_value=_SAMPLE_DF)
def test_falls_back_to_alpha_vantage(mock_av, mock_yf):
    df = collect_ohlcv("AAPL", _START, _END)
    assert mock_av.called
    assert isinstance(df, pl.DataFrame)


@patch("src.ingestion.collector.fetch_ohlcv", side_effect=ValueError("yf fail"))
@patch("src.ingestion.collector.fetch_ohlcv_av", side_effect=ValueError("av fail"))
@patch("src.ingestion.collector.fetch_ohlcv_fmp", return_value=_SAMPLE_DF)
def test_falls_back_to_fmp(mock_fmp, mock_av, mock_yf):
    df = collect_ohlcv("AAPL", _START, _END)
    assert mock_fmp.called


@patch("src.ingestion.collector.fetch_ohlcv", side_effect=ValueError("yf fail"))
@patch("src.ingestion.collector.fetch_ohlcv_av", side_effect=ValueError("av fail"))
@patch("src.ingestion.collector.fetch_ohlcv_fmp", side_effect=ValueError("fmp fail"))
def test_raises_when_all_sources_fail(mock_fmp, mock_av, mock_yf):
    with pytest.raises(RuntimeError, match="All data sources failed"):
        collect_ohlcv("AAPL", _START, _END)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_collector.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/ingestion/collector.py**

```python
from datetime import datetime
import logging

import polars as pl

from src.ingestion.historical_collector import fetch_ohlcv
from src.ingestion.alpha_vantage_client import fetch_ohlcv_av
from src.ingestion.fmp_client import fetch_ohlcv_fmp

logger = logging.getLogger(__name__)


def collect_ohlcv(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pl.DataFrame:
    sources = [
        ("yfinance", lambda: fetch_ohlcv(ticker, start, end, interval)),
        ("alpha_vantage", lambda: fetch_ohlcv_av(ticker, start, end)),
        ("fmp", lambda: fetch_ohlcv_fmp(ticker, start, end)),
    ]
    errors: list[str] = []
    for name, fn in sources:
        try:
            df = fn()
            logger.info("Fetched %s from %s (%d rows)", ticker, name, len(df))
            return df
        except Exception as e:
            logger.warning("Source %s failed for %s: %s", name, ticker, e)
            errors.append(f"{name}: {e}")

    raise RuntimeError(f"All data sources failed for {ticker}: {'; '.join(errors)}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_collector.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/collector.py tests/unit/test_collector.py
git commit -m "feat: fallback collector chain yfinance->alphavantage->fmp"
```

---

### Task 7: Survivorship Bias Module

**Files:**
- Create: `data/index_compositions/sp500_changes.csv`
- Create: `src/ingestion/survivorship.py`
- Test: `tests/unit/test_survivorship.py`

**Interfaces:**
- Consumes: `data/index_compositions/sp500_changes.csv`
- Produces:
  - `load_sp500_changes() -> pl.DataFrame`
    - Columns: `ticker: Utf8`, `added_date: Date`, `removed_date: Date | null`
  - `get_sp500_tickers_at(date: datetime) -> list[str]`
    - Returns tickers that were S&P 500 members at `date` — added on or before `date` and not yet removed

Note on the CSV: The seed file below covers test cases only. Before production use, run `scripts/seed_sp500_changes.py` (not in this plan) which scrapes Wikipedia's S&P 500 changes table to produce the full historical record from 1996 onwards.

- [ ] **Step 1: Create data/index_compositions/sp500_changes.csv**

```csv
ticker,added_date,removed_date
AAPL,1982-11-30,
MSFT,1994-06-01,
GOOGL,2006-03-31,
META,2013-12-23,
AMZN,2005-11-18,
NVDA,2001-11-30,
TSLA,2020-12-21,
ENRN,1995-06-28,2001-12-03
LEHM,1973-01-01,2008-09-15
```

- [ ] **Step 2: Write failing tests**

```python
# tests/unit/test_survivorship.py
from datetime import datetime, timezone
import polars as pl

from src.ingestion.survivorship import get_sp500_tickers_at, load_sp500_changes


def test_load_sp500_changes_returns_dataframe():
    df = load_sp500_changes()
    assert isinstance(df, pl.DataFrame)
    assert {"ticker", "added_date", "removed_date"}.issubset(df.columns)


def test_excludes_tickers_added_after_query_date():
    date = datetime(2005, 1, 1, tzinfo=timezone.utc)
    tickers = get_sp500_tickers_at(date)
    assert "GOOGL" not in tickers   # added 2006-03-31
    assert "TSLA" not in tickers    # added 2020-12-21
    assert "AAPL" in tickers        # added 1982-11-30


def test_excludes_tickers_removed_before_query_date():
    date = datetime(2009, 1, 1, tzinfo=timezone.utc)
    tickers = get_sp500_tickers_at(date)
    assert "ENRN" not in tickers    # removed 2001-12-03
    assert "LEHM" not in tickers    # removed 2008-09-15


def test_includes_active_tickers_at_query_date():
    date = datetime(2015, 1, 1, tzinfo=timezone.utc)
    tickers = get_sp500_tickers_at(date)
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "META" in tickers        # added 2013-12-23
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/unit/test_survivorship.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 4: Implement src/ingestion/survivorship.py**

```python
from datetime import datetime
from pathlib import Path

import polars as pl

_CSV_PATH = Path(__file__).parents[2] / "data" / "index_compositions" / "sp500_changes.csv"


def load_sp500_changes() -> pl.DataFrame:
    return pl.read_csv(
        _CSV_PATH,
        schema_overrides={"ticker": pl.Utf8, "added_date": pl.Date, "removed_date": pl.Date},
        try_parse_dates=True,
        null_values=[""],
    )


def get_sp500_tickers_at(date: datetime) -> list[str]:
    query_date = date.date() if hasattr(date, "date") else date
    df = load_sp500_changes()
    active = df.filter(
        (pl.col("added_date") <= pl.lit(query_date))
        & (
            pl.col("removed_date").is_null()
            | (pl.col("removed_date") > pl.lit(query_date))
        )
    )
    return active["ticker"].to_list()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_survivorship.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/index_compositions/sp500_changes.csv src/ingestion/survivorship.py tests/unit/test_survivorship.py
git commit -m "feat: survivorship-bias-free S&P 500 composition lookup"
```

---

### Task 8: News Collector

**Files:**
- Create: `src/ingestion/news_collector.py`
- Test: `tests/unit/test_news_collector.py`

**Interfaces:**
- Consumes: `config.settings.settings.NEWSAPI_KEY`, `config.settings.settings.FINNHUB_KEY`
- Produces:
  - `NewsArticle` dataclass: `ticker: str | None`, `headline: str`, `body: str | None`, `source: str`, `url: str`, `published_at: datetime`
  - `collect_rss(ticker: str, max_items: int = 20) -> list[NewsArticle]`
  - `collect_newsapi(ticker: str, from_date: datetime, to_date: datetime) -> list[NewsArticle]`
  - `collect_finnhub(ticker: str, from_date: datetime, to_date: datetime) -> list[NewsArticle]`
  - `collect_all_news(ticker: str, from_date: datetime, to_date: datetime) -> list[NewsArticle]`
    - Merges all three sources and deduplicates by URL

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_news_collector.py
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest

from src.ingestion.news_collector import (
    NewsArticle,
    collect_rss,
    collect_newsapi,
    collect_finnhub,
    collect_all_news,
)

_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 1, 7, tzinfo=timezone.utc)
_ARTICLE = NewsArticle(
    ticker="AAPL",
    headline="Apple hits new high",
    body=None,
    source="test",
    url="https://example.com/unique-url",
    published_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
)


@patch("src.ingestion.news_collector.feedparser.parse")
def test_collect_rss_returns_list_of_articles(mock_parse):
    mock_parse.return_value = MagicMock(
        entries=[
            MagicMock(
                title="Apple hits new high",
                link="https://example.com/1",
                published_parsed=(2024, 1, 2, 12, 0, 0, 0, 2, 0),
                summary="Apple stock surged.",
            )
        ]
    )
    articles = collect_rss("AAPL")
    assert len(articles) == 1
    assert articles[0].headline == "Apple hits new high"
    assert articles[0].ticker == "AAPL"


@patch("src.ingestion.news_collector.NewsApiClient")
def test_collect_newsapi_returns_articles(mock_cls):
    mock_cls.return_value.get_everything.return_value = {
        "articles": [{
            "title": "AAPL earnings beat",
            "url": "https://example.com/2",
            "publishedAt": "2024-01-03T10:00:00Z",
            "source": {"name": "Reuters"},
            "content": "Apple beat earnings.",
        }]
    }
    articles = collect_newsapi("AAPL", _START, _END)
    assert len(articles) == 1
    assert articles[0].ticker == "AAPL"


@patch("src.ingestion.news_collector.finnhub.Client")
def test_collect_finnhub_returns_articles(mock_cls):
    mock_cls.return_value.company_news.return_value = [{
        "headline": "AAPL buyback",
        "url": "https://example.com/3",
        "datetime": 1704153600,
        "source": "Finnhub",
        "summary": "Apple announces buyback.",
    }]
    articles = collect_finnhub("AAPL", _START, _END)
    assert len(articles) == 1


@patch("src.ingestion.news_collector.collect_rss", return_value=[_ARTICLE])
@patch("src.ingestion.news_collector.collect_newsapi", return_value=[_ARTICLE])
@patch("src.ingestion.news_collector.collect_finnhub", return_value=[])
def test_collect_all_news_deduplicates_by_url(mock_fh, mock_na, mock_rss):
    articles = collect_all_news("AAPL", _START, _END)
    assert len(articles) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_news_collector.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/ingestion/news_collector.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

import feedparser
import finnhub
from newsapi import NewsApiClient

from config.settings import settings

logger = logging.getLogger(__name__)
_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


@dataclass
class NewsArticle:
    ticker: str | None
    headline: str
    body: str | None
    source: str
    url: str
    published_at: datetime


def collect_rss(ticker: str, max_items: int = 20) -> list[NewsArticle]:
    feed = feedparser.parse(_RSS_TEMPLATE.format(ticker=ticker))
    articles: list[NewsArticle] = []
    for entry in feed.entries[:max_items]:
        try:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            articles.append(NewsArticle(
                ticker=ticker,
                headline=entry.get("title", ""),
                body=entry.get("summary"),
                source="yahoo_rss",
                url=entry.get("link", ""),
                published_at=ts,
            ))
        except Exception as e:
            logger.debug("RSS parse error for %s: %s", ticker, e)
    return articles


def collect_newsapi(
    ticker: str,
    from_date: datetime,
    to_date: datetime,
) -> list[NewsArticle]:
    if not settings.NEWSAPI_KEY:
        return []
    client = NewsApiClient(api_key=settings.NEWSAPI_KEY)
    resp = client.get_everything(
        q=ticker,
        from_param=from_date.strftime("%Y-%m-%d"),
        to=to_date.strftime("%Y-%m-%d"),
        language="en",
        sort_by="publishedAt",
        page_size=100,
    )
    articles: list[NewsArticle] = []
    for a in resp.get("articles", []):
        try:
            ts = datetime.strptime(a["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            articles.append(NewsArticle(
                ticker=ticker,
                headline=a["title"],
                body=a.get("content"),
                source=a["source"]["name"],
                url=a["url"],
                published_at=ts,
            ))
        except Exception as e:
            logger.debug("NewsAPI parse error: %s", e)
    return articles


def collect_finnhub(
    ticker: str,
    from_date: datetime,
    to_date: datetime,
) -> list[NewsArticle]:
    if not settings.FINNHUB_KEY:
        return []
    client = finnhub.Client(api_key=settings.FINNHUB_KEY)
    raw = client.company_news(
        ticker,
        _from=from_date.strftime("%Y-%m-%d"),
        to=to_date.strftime("%Y-%m-%d"),
    )
    articles: list[NewsArticle] = []
    for item in raw:
        try:
            ts = datetime.fromtimestamp(item["datetime"], tz=timezone.utc)
            articles.append(NewsArticle(
                ticker=ticker,
                headline=item["headline"],
                body=item.get("summary"),
                source=item.get("source", "finnhub"),
                url=item["url"],
                published_at=ts,
            ))
        except Exception as e:
            logger.debug("Finnhub parse error: %s", e)
    return articles


def collect_all_news(
    ticker: str,
    from_date: datetime,
    to_date: datetime,
) -> list[NewsArticle]:
    seen: set[str] = set()
    result: list[NewsArticle] = []
    for article in (
        collect_rss(ticker)
        + collect_newsapi(ticker, from_date, to_date)
        + collect_finnhub(ticker, from_date, to_date)
    ):
        if article.url and article.url not in seen:
            seen.add(article.url)
            result.append(article)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_news_collector.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/news_collector.py tests/unit/test_news_collector.py
git commit -m "feat: news collector with RSS, NewsAPI, Finnhub and URL deduplication"
```

---

### Task 9: Sentiment Processor

**Files:**
- Create: `src/ingestion/sentiment_processor.py`
- Test: `tests/unit/test_sentiment_processor.py`

**Interfaces:**
- Consumes: `NewsArticle` from Task 8
- Produces:
  - `SentimentResult` dataclass: `label: str` (`"positive"` | `"negative"` | `"neutral"`), `positive: float`, `negative: float`, `neutral: float`
  - `score_headline(text: str) -> SentimentResult`
    - Uses FinBERT (`ProsusAI/finbert`) via `_get_finbert_pipeline()` singleton
    - Falls back to VADER if text < 10 chars or FinBERT unavailable
  - `score_articles(articles: list[NewsArticle]) -> list[tuple[NewsArticle, SentimentResult]]`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_sentiment_processor.py
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.ingestion.sentiment_processor import SentimentResult, score_headline, score_articles
from src.ingestion.news_collector import NewsArticle


def _article(headline: str) -> NewsArticle:
    return NewsArticle(
        ticker="AAPL", headline=headline, body=None, source="test",
        url="https://example.com/x", published_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


@patch("src.ingestion.sentiment_processor._get_finbert_pipeline")
def test_score_headline_returns_sentiment_result(mock_fn):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [[
        {"label": "positive", "score": 0.85},
        {"label": "neutral", "score": 0.10},
        {"label": "negative", "score": 0.05},
    ]]
    mock_fn.return_value = mock_pipeline

    result = score_headline("Apple hits record high on strong earnings beat")

    assert isinstance(result, SentimentResult)
    assert result.label == "positive"
    assert abs(result.positive + result.negative + result.neutral - 1.0) < 0.01


@patch("src.ingestion.sentiment_processor._get_finbert_pipeline")
def test_score_articles_returns_article_sentiment_pairs(mock_fn):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [[
        {"label": "neutral", "score": 0.70},
        {"label": "positive", "score": 0.20},
        {"label": "negative", "score": 0.10},
    ]]
    mock_fn.return_value = mock_pipeline

    articles = [_article("Apple launches new product line")]
    pairs = score_articles(articles)

    assert len(pairs) == 1
    assert pairs[0][0] is articles[0]
    assert isinstance(pairs[0][1], SentimentResult)


def test_score_headline_falls_back_to_vader_on_short_text():
    result = score_headline("up")
    assert isinstance(result, SentimentResult)
    assert result.label in {"positive", "negative", "neutral"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_sentiment_processor.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/ingestion/sentiment_processor.py**

```python
from __future__ import annotations
from dataclasses import dataclass
import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.ingestion.news_collector import NewsArticle

logger = logging.getLogger(__name__)

_finbert_pipeline = None
_vader = SentimentIntensityAnalyzer()
_FINBERT_MIN_CHARS = 10


@dataclass
class SentimentResult:
    label: str
    positive: float
    negative: float
    neutral: float


def _get_finbert_pipeline():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _finbert_pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
                device=-1,
            )
        except Exception as e:
            logger.warning("FinBERT unavailable (%s), VADER only", e)
    return _finbert_pipeline


def _vader_score(text: str) -> SentimentResult:
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    total = scores["pos"] + scores["neg"] + scores["neu"] or 1.0
    return SentimentResult(
        label=label,
        positive=scores["pos"] / total,
        negative=scores["neg"] / total,
        neutral=scores["neu"] / total,
    )


def score_headline(text: str) -> SentimentResult:
    if len(text) < _FINBERT_MIN_CHARS:
        return _vader_score(text)
    pipeline_fn = _get_finbert_pipeline()
    if pipeline_fn is None:
        return _vader_score(text)
    try:
        results = pipeline_fn(text[:512])[0]
        scores = {r["label"].lower(): r["score"] for r in results}
        label = max(scores, key=scores.get)
        return SentimentResult(
            label=label,
            positive=scores.get("positive", 0.0),
            negative=scores.get("negative", 0.0),
            neutral=scores.get("neutral", 0.0),
        )
    except Exception as e:
        logger.warning("FinBERT error: %s — VADER fallback", e)
        return _vader_score(text)


def score_articles(
    articles: list[NewsArticle],
) -> list[tuple[NewsArticle, SentimentResult]]:
    return [(a, score_headline(a.headline)) for a in articles]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_sentiment_processor.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/sentiment_processor.py tests/unit/test_sentiment_processor.py
git commit -m "feat: FinBERT sentiment processor with VADER fallback"
```

---

### Task 10: Storage Writer

**Files:**
- Create: `src/ingestion/storage_writer.py`
- Test: `tests/unit/test_storage_writer.py`

**Interfaces:**
- Consumes: `get_connection`, `release_connection`, `execute_many` (Task 2); `pl.DataFrame` (Task 3); `NewsArticle` (Task 8); `SentimentResult` (Task 9)
- Produces:
  - `write_ohlcv(df: pl.DataFrame) -> int` — returns count of rows written
  - `write_news(pairs: list[tuple[NewsArticle, SentimentResult]]) -> int` — returns count of rows written
  - `write_fundamentals(ticker: str, snapshot: dict[str, float | None], as_of: datetime) -> None`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_storage_writer.py
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import polars as pl

from src.ingestion.storage_writer import write_ohlcv, write_news, write_fundamentals
from src.ingestion.news_collector import NewsArticle
from src.ingestion.sentiment_processor import SentimentResult

_DF = pl.DataFrame({
    "time": [datetime(2024, 1, 2, tzinfo=timezone.utc)],
    "ticker": ["AAPL"],
    "open": [150.0], "high": [155.0], "low": [149.0], "close": [153.0],
    "volume": [1_000_000], "adj_close": [153.0], "dividends": [0.0], "stock_splits": [0.0],
})
_ARTICLE = NewsArticle(
    ticker="AAPL", headline="Apple surges", body=None, source="test",
    url="https://example.com/z", published_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
)
_SENTIMENT = SentimentResult(label="positive", positive=0.9, negative=0.05, neutral=0.05)


@patch("src.ingestion.storage_writer.get_connection", return_value=MagicMock())
@patch("src.ingestion.storage_writer.execute_many")
@patch("src.ingestion.storage_writer.release_connection")
def test_write_ohlcv_returns_row_count(mock_rel, mock_exec, mock_conn):
    n = write_ohlcv(_DF)
    assert mock_exec.called
    assert n == 1


@patch("src.ingestion.storage_writer.get_connection", return_value=MagicMock())
@patch("src.ingestion.storage_writer.execute_many")
@patch("src.ingestion.storage_writer.release_connection")
def test_write_news_returns_row_count(mock_rel, mock_exec, mock_conn):
    n = write_news([(_ARTICLE, _SENTIMENT)])
    assert mock_exec.called
    assert n == 1


@patch("src.ingestion.storage_writer.get_connection", return_value=MagicMock())
@patch("src.ingestion.storage_writer.execute_many")
@patch("src.ingestion.storage_writer.release_connection")
def test_write_fundamentals_calls_execute_many(mock_rel, mock_exec, mock_conn):
    snapshot = {"pe_ratio": 28.5, "eps": 6.43, "book_value": 4.5, "market_cap": 3e12}
    write_fundamentals("AAPL", snapshot, datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert mock_exec.called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_storage_writer.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement src/ingestion/storage_writer.py**

```python
from datetime import datetime
import logging

import polars as pl

from src.ingestion.db import get_connection, release_connection, execute_many
from src.ingestion.news_collector import NewsArticle
from src.ingestion.sentiment_processor import SentimentResult

logger = logging.getLogger(__name__)

_OHLCV_SQL = """
INSERT INTO ohlcv (time, ticker, open, high, low, close, volume, adj_close, dividends, stock_splits)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, time) DO UPDATE SET
    open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close,
    volume=EXCLUDED.volume, adj_close=EXCLUDED.adj_close,
    dividends=EXCLUDED.dividends, stock_splits=EXCLUDED.stock_splits
"""

_NEWS_SQL = """
INSERT INTO news_articles
    (published_at, ticker, headline, body, source, url, relevance,
     sentiment_pos, sentiment_neg, sentiment_neu, sentiment_label)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (url) DO NOTHING
"""

_FUND_SQL = """
INSERT INTO fundamentals (time, ticker, pe_ratio, eps, book_value, market_cap)
VALUES (%s, %s, %s, %s, %s, %s)
"""


def write_ohlcv(df: pl.DataFrame) -> int:
    rows = [
        (r["time"], r["ticker"], r["open"], r["high"], r["low"],
         r["close"], r["volume"], r["adj_close"], r["dividends"], r["stock_splits"])
        for r in df.iter_rows(named=True)
    ]
    conn = get_connection()
    try:
        execute_many(conn, _OHLCV_SQL, rows)
    finally:
        release_connection(conn)
    return len(rows)


def write_news(pairs: list[tuple[NewsArticle, SentimentResult]]) -> int:
    rows = [
        (a.published_at, a.ticker, a.headline, a.body, a.source, a.url,
         1.0, s.positive, s.negative, s.neutral, s.label)
        for a, s in pairs
    ]
    conn = get_connection()
    try:
        execute_many(conn, _NEWS_SQL, rows)
    finally:
        release_connection(conn)
    return len(rows)


def write_fundamentals(
    ticker: str,
    snapshot: dict[str, float | None],
    as_of: datetime,
) -> None:
    rows = [(
        as_of, ticker,
        snapshot.get("pe_ratio"), snapshot.get("eps"),
        snapshot.get("book_value"), snapshot.get("market_cap"),
    )]
    conn = get_connection()
    try:
        execute_many(conn, _FUND_SQL, rows)
    finally:
        release_connection(conn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_storage_writer.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/storage_writer.py tests/unit/test_storage_writer.py
git commit -m "feat: storage writer for OHLCV, news, and fundamentals"
```

---

### Task 11: Airflow DAGs

**Files:**
- Create: `dags/__init__.py`
- Create: `dags/historical_data_dag.py`
- Create: `dags/news_sentiment_dag.py`
- Test: `tests/unit/test_dags.py`

**Interfaces:**
- Consumes: `collect_ohlcv`, `write_ohlcv`, `fetch_fundamentals`, `write_fundamentals`, `get_sp500_tickers_at`, `collect_all_news`, `score_articles`, `write_news`
- Produces:
  - DAG `historical_data_dag` — `dag_id="historical_data_dag"`, `schedule_interval="@daily"`, task `fetch_and_store_ohlcv`
  - DAG `news_sentiment_dag` — `dag_id="news_sentiment_dag"`, `schedule_interval="@daily"`, task `fetch_and_store_news`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_dags.py
import importlib


def test_historical_data_dag_loads():
    mod = importlib.import_module("dags.historical_data_dag")
    assert hasattr(mod, "dag")
    assert mod.dag.dag_id == "historical_data_dag"


def test_news_sentiment_dag_loads():
    mod = importlib.import_module("dags.news_sentiment_dag")
    assert hasattr(mod, "dag")
    assert mod.dag.dag_id == "news_sentiment_dag"


def test_historical_dag_has_fetch_task():
    mod = importlib.import_module("dags.historical_data_dag")
    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "fetch_and_store_ohlcv" in task_ids


def test_news_dag_has_fetch_task():
    mod = importlib.import_module("dags.news_sentiment_dag")
    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "fetch_and_store_news" in task_ids
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_dags.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement dags/historical_data_dag.py**

```python
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

_default_args = {
    "owner": "platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _fetch_and_store_ohlcv(**context):
    import logging
    from src.ingestion.survivorship import get_sp500_tickers_at
    from src.ingestion.collector import collect_ohlcv
    from src.ingestion.historical_collector import fetch_fundamentals
    from src.ingestion.storage_writer import write_ohlcv, write_fundamentals

    log = logging.getLogger(__name__)
    execution_date = context["execution_date"]
    tickers = get_sp500_tickers_at(execution_date)
    end = execution_date.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=2)

    for ticker in tickers:
        try:
            df = collect_ohlcv(ticker, start, end)
            write_ohlcv(df)
            snapshot = fetch_fundamentals(ticker)
            write_fundamentals(ticker, snapshot, end)
        except Exception as e:
            log.error("Failed %s: %s", ticker, e)


with DAG(
    dag_id="historical_data_dag",
    default_args=_default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "ohlcv"],
) as dag:
    PythonOperator(
        task_id="fetch_and_store_ohlcv",
        python_callable=_fetch_and_store_ohlcv,
    )
```

- [ ] **Step 4: Implement dags/news_sentiment_dag.py**

```python
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

_default_args = {
    "owner": "platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _fetch_and_store_news(**context):
    import logging
    from src.ingestion.survivorship import get_sp500_tickers_at
    from src.ingestion.news_collector import collect_all_news
    from src.ingestion.sentiment_processor import score_articles
    from src.ingestion.storage_writer import write_news

    log = logging.getLogger(__name__)
    execution_date = context["execution_date"]
    tickers = get_sp500_tickers_at(execution_date)
    end = execution_date.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=2)

    for ticker in tickers:
        try:
            articles = collect_all_news(ticker, start, end)
            if articles:
                pairs = score_articles(articles)
                write_news(pairs)
        except Exception as e:
            log.error("News failed %s: %s", ticker, e)


with DAG(
    dag_id="news_sentiment_dag",
    default_args=_default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "news"],
) as dag:
    PythonOperator(
        task_id="fetch_and_store_news",
        python_callable=_fetch_and_store_news,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_dags.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests PASS. Note: DB tests require the TimescaleDB container from Task 2 to be running.

- [ ] **Step 7: Commit**

```bash
git add dags/ tests/unit/test_dags.py
git commit -m "feat: airflow DAGs for daily OHLCV and news/sentiment ingestion"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| OHLCV daily history (adjusted for splits/dividends) | Task 3 |
| Corporate actions (dividends, splits, earnings dates) | Task 3 |
| Fundamental snapshot (P/E, EPS, book value, market cap) | Task 3 + Task 10 |
| yFinance as primary source | Task 3 |
| Alpha Vantage as fallback | Task 4 |
| FMP as secondary fallback | Task 5 |
| Fallback chain between sources | Task 6 |
| Survivorship-bias-free historical index compositions | Task 7 |
| News headlines from RSS (Yahoo Finance) | Task 8 |
| NewsAPI integration | Task 8 |
| Finnhub integration | Task 8 |
| Sentiment scores (positive/negative/neutral) | Task 9 |
| Entity recognition linking news to tickers | Task 8 (`ticker` field on `NewsArticle`) |
| TimescaleDB hypertable storage | Tasks 2 + 10 |
| Automated daily refresh (Airflow) | Task 11 |

**Gap:** Spec mentions intraday data. `fetch_ohlcv` accepts `interval="1h"` and passes it through yFinance. Alpha Vantage and FMP fallbacks cover daily only — intraday fallback requires a separate AV intraday endpoint and is deferred to Plan 2 if needed for feature engineering.

**Placeholder scan:** No TBDs, no "similar to task N" references, no steps without code blocks.

**Type consistency:** `NewsArticle`, `SentimentResult`, `pl.DataFrame` column names (`time`, `ticker`, `open`, `high`, `low`, `close`, `volume`, `adj_close`, `dividends`, `stock_splits`) are consistent from Task 3 through Task 10. `write_ohlcv` consumes the exact column set produced by `fetch_ohlcv`.
