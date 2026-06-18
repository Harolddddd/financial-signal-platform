import importlib


def test_model_retrain_dag_loads():
    mod = importlib.import_module("dags.model_retrain_dag")
    assert hasattr(mod, "dag")
    assert mod.dag.dag_id == "model_retrain_dag"


def test_dag_has_expected_tasks():
    mod = importlib.import_module("dags.model_retrain_dag")
    task_ids = {t.task_id for t in mod.dag.tasks}
    assert "tune_and_train_models" in task_ids
    assert "evaluate_and_register" in task_ids


def test_evaluate_depends_on_tune():
    mod = importlib.import_module("dags.model_retrain_dag")
    dag = mod.dag
    tune_task = dag.get_task("tune_and_train_models")
    eval_task = dag.get_task("evaluate_and_register")
    assert eval_task.task_id in {d.task_id for d in tune_task.downstream_list}
