import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from config.settings import settings

_pool: pg_pool.SimpleConnectionPool | None = None


def _get_pool() -> pg_pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.SimpleConnectionPool(1, 10, settings.DATABASE_URL)
    return _pool


def get_connection() -> psycopg2.extensions.connection:
    return _get_pool().getconn()


def release_connection(conn: psycopg2.extensions.connection) -> None:
    _get_pool().putconn(conn)


def execute_many(
    conn: psycopg2.extensions.connection,
    sql: str,
    rows: list[tuple],
) -> None:
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=1000)
    conn.commit()
