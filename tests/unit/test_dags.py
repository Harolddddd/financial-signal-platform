import importlib


def test_historical_data_dag_loads():
    mod = importlib.import_module("dags.historical_data_dag")
    assert hasattr(mod, "dag")
    assert mod.dag.dag_id == "historical_data_dag"


def test_news_sentiment_dag_loads():
    mod = importlib.import_module("dags.news_sentiment_dag")
    assert hasattr(mod, "dag")
    assert mod.dag.dag_id == "news_sentiment_dag"


def test_historical_dag_has_fetch_task():
    mod = importlib.import_module("dags.historical_data_dag")
    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "fetch_and_store_ohlcv" in task_ids


def test_news_dag_has_fetch_task():
    mod = importlib.import_module("dags.news_sentiment_dag")
    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "fetch_and_store_news" in task_ids
