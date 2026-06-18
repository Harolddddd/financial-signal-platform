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
