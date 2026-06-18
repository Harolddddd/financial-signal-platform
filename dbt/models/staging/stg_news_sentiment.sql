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
