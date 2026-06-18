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
