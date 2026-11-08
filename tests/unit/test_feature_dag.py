import importlib

import pytest

# Legacy Airflow DAGs (the servers run systemd timers). Skip when the installed
# Airflow no longer has the API these DAG files were written for.
try:
    from airflow.sdk import ObjectStoragePath  # noqa: F401
except ImportError:
    pytest.skip("Airflow API used by the legacy DAGs is not available", allow_module_level=True)


def test_feature_engineering_dag_loads():
    mod = importlib.import_module("dags.feature_engineering_dag")
    assert hasattr(mod, "dag")
    assert mod.dag.dag_id == "feature_engineering_dag"


def test_dag_has_expected_tasks():
    mod = importlib.import_module("dags.feature_engineering_dag")
    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "compute_and_store_features" in task_ids
    assert "export_to_parquet" in task_ids


def test_export_depends_on_compute():
    mod = importlib.import_module("dags.feature_engineering_dag")
    dag = mod.dag
    compute = dag.get_task("compute_and_store_features")
    export = dag.get_task("export_to_parquet")
    assert export.task_id in {d.task_id for d in compute.downstream_list}
