from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

_default_args = {
    "owner": "platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

_PARQUET_DIR = Path("/opt/airflow/data/features")


def _compute_and_store(**context):
    import logging
    from src.ingestion.survivorship import get_sp500_tickers_at
    from src.features.feature_store import build_features, write_features

    log = logging.getLogger(__name__)
    execution_date = context["execution_date"]
    tickers = get_sp500_tickers_at(execution_date)
    end = execution_date.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=365 * 2)

    for ticker in tickers:
        try:
            df = build_features(ticker, start, end)
            if not df.is_empty():
                write_features(df)
                context["ti"].xcom_push(key=ticker, value=len(df))
        except Exception as e:
            log.error("Feature build failed %s: %s", ticker, e)


def _export_parquet(**context):
    import logging
    from src.ingestion.survivorship import get_sp500_tickers_at
    from src.features.feature_store import build_features, export_parquet

    log = logging.getLogger(__name__)
    execution_date = context["execution_date"]
    tickers = get_sp500_tickers_at(execution_date)
    end = execution_date.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=365 * 2)

    for ticker in tickers:
        try:
            df = build_features(ticker, start, end)
            if not df.is_empty():
                path = export_parquet(df, ticker, _PARQUET_DIR)
                log.info("Exported %s → %s", ticker, path)
        except Exception as e:
            log.error("Parquet export failed %s: %s", ticker, e)


with DAG(
    dag_id="feature_engineering_dag",
    default_args=_default_args,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features"],
) as dag:
    compute = PythonOperator(
        task_id="compute_and_store_features",
        python_callable=_compute_and_store,
    )
    export = PythonOperator(
        task_id="export_to_parquet",
        python_callable=_export_parquet,
    )
    compute >> export
