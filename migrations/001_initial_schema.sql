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
