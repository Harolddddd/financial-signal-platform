from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

_default_args = {
    "owner": "platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

_PARQUET_DIR  = Path("/opt/airflow/data/features")
_REGISTRY_DIR = Path("/opt/airflow/data/registry")
_FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "atr_14", "hist_vol_21",
    "sent_pos_avg_3d", "sent_pos_avg_5d", "sent_pos_avg_10d",
    "sent_pos_mom_3d", "news_vol_spike", "rel_strength_spy", "vix_level",
]

_MODEL_ZOO = [
    ("logistic_regression", "src.models.zoo.logistic_regression.LogisticRegressionClassifier"),
    ("random_forest",        "src.models.zoo.random_forest.RandomForestClassifier_"),
    ("xgboost",              "src.models.zoo.xgboost_model.XGBoostClassifier"),
    ("lightgbm",             "src.models.zoo.lightgbm_model.LightGBMClassifier"),
    ("svm",                  "src.models.zoo.svm_model.SVMClassifier"),
    ("naive_bayes",          "src.models.zoo.naive_bayes.NaiveBayesClassifier"),
    ("mlp",                  "src.models.zoo.mlp_model.MLPClassifier_"),
    ("lstm",                 "src.models.zoo.lstm_model.LSTMClassifier"),
]


def _tune_and_train(**context):
    import importlib
    import logging
    from src.features.duckdb_client import load_training_data
    from src.models.tuner import tune

    log = logging.getLogger(__name__)
    df = load_training_data(_PARQUET_DIR)
    df_clean = df.drop_nulls(subset=_FEATURE_COLS + ["label"])

    n = len(df_clean)
    split = int(n * 0.8)
    train_df = df_clean[:split]
    val_df = df_clean[split:]

    X_train = train_df.select(_FEATURE_COLS).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_val = val_df.select(_FEATURE_COLS).to_numpy()
    y_val = val_df["label"].to_numpy()

    best_params: dict[str, dict] = {}
    for model_name, model_path in _MODEL_ZOO:
        try:
            module_path, class_name = model_path.rsplit(".", 1)
            ModelClass = getattr(importlib.import_module(module_path), class_name)
            params = tune(model_name, ModelClass, X_train, y_train, X_val, y_val, n_trials=50)
            best_params[model_name] = {"path": model_path, "params": params}
            log.info("Tuned %s: %s", model_name, params)
        except Exception as e:
            log.error("Tune failed %s: %s", model_name, e)

    context["ti"].xcom_push(key="best_params", value=best_params)


def _evaluate_and_register(**context):
    import importlib
    import logging
    from src.features.duckdb_client import load_training_data
    from src.models.trainer import walk_forward_train
    from src.models.registry import save_model

    log = logging.getLogger(__name__)
    best_params = context["ti"].xcom_pull(key="best_params", task_ids="tune_and_train_models") or {}
    df = load_training_data(_PARQUET_DIR)

    for model_name, info in best_params.items():
        try:
            module_path, class_name = info["path"].rsplit(".", 1)
            ModelClass = getattr(importlib.import_module(module_path), class_name)
            clf = ModelClass(**info["params"])
            result = walk_forward_train(
                df, clf, _FEATURE_COLS,
                train_window_days=500, test_window_days=21, step_days=21,
            )
            last_eval = result.folds[-1].evaluation
            save_model(clf, last_eval, info["params"], _FEATURE_COLS, _REGISTRY_DIR)
            log.info("Registered %s — precision_buy=%.3f", model_name, last_eval.precision_buy)
        except Exception as e:
            log.error("Register failed %s: %s", model_name, e)


with DAG(
    dag_id="model_retrain_dag",
    default_args=_default_args,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["models", "training"],
) as dag:
    tune_task = PythonOperator(
        task_id="tune_and_train_models",
        python_callable=_tune_and_train,
    )
    eval_task = PythonOperator(
        task_id="evaluate_and_register",
        python_callable=_evaluate_and_register,
    )
    tune_task >> eval_task
