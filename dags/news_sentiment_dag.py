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
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "news"],
) as dag:
    PythonOperator(
        task_id="fetch_and_store_news",
        python_callable=_fetch_and_store_news,
    )
