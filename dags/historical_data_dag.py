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
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "ohlcv"],
) as dag:
    PythonOperator(
        task_id="fetch_and_store_ohlcv",
        python_callable=_fetch_and_store_ohlcv,
    )
