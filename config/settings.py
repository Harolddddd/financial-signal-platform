from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str = "postgresql://platform:platform@localhost:5432/financial"

    ALPHA_VANTAGE_KEY: str = ""
    FMP_KEY: str = ""
    NEWSAPI_KEY: str = ""
    FINNHUB_KEY: str = ""

    STOCK_UNIVERSE: str = "sp500"
    HISTORICAL_DAYS: int = 3650


settings = Settings()
