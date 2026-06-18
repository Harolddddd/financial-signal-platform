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
