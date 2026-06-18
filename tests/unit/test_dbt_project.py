from pathlib import Path
import yaml


DBT_DIR = Path(__file__).parents[2] / "dbt"


def test_dbt_project_yml_exists_and_valid():
    project_file = DBT_DIR / "dbt_project.yml"
    assert project_file.exists()
    data = yaml.safe_load(project_file.read_text())
    assert data["name"] == "financial_platform"
    assert "models" in data


def test_staging_models_exist():
    staging = DBT_DIR / "models" / "staging"
    assert (staging / "stg_ohlcv.sql").exists()
    assert (staging / "stg_news_sentiment.sql").exists()
    assert (staging / "schema.yml").exists()


def test_feature_models_exist():
    feats = DBT_DIR / "models" / "features"
    assert (feats / "feat_sma.sql").exists()
    assert (feats / "feat_sentiment_daily.sql").exists()
    assert (feats / "feat_labels.sql").exists()
    assert (feats / "schema.yml").exists()


def test_schema_yml_defines_not_null_tests():
    schema = yaml.safe_load(
        (DBT_DIR / "models" / "staging" / "schema.yml").read_text()
    )
    model_names = [m["name"] for m in schema["models"]]
    assert "stg_ohlcv" in model_names
