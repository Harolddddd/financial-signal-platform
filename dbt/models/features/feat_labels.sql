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
