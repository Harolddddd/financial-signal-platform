SELECT
    DATE(published_at)              AS date,
    ticker                          AS ticker,
    AVG(sentiment_pos)              AS avg_pos,
    AVG(sentiment_neg)              AS avg_neg,
    COUNT(*)                        AS article_count
FROM {{ ref('stg_news_sentiment') }}
GROUP BY DATE(published_at), ticker
ORDER BY date, ticker
