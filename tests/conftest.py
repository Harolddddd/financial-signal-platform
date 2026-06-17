import pytest
import psycopg2
from config.settings import settings


@pytest.fixture(scope="session")
def db_conn():
    conn = psycopg2.connect(settings.DATABASE_URL)
    yield conn
    conn.close()


@pytest.fixture
def sample_tickers() -> list[str]:
    return ["AAPL", "MSFT", "GOOGL"]
