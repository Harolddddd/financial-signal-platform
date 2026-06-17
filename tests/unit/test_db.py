import pytest
from unittest.mock import Mock, patch, MagicMock, call
from src.ingestion.db import get_connection, execute_many, release_connection


@patch('src.ingestion.db._pool', None)
def test_get_connection_returns_open_connection():
    """Test that get_connection returns an open connection."""
    # Mock the pool class
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.closed = 0
    mock_pool.getconn.return_value = mock_conn

    with patch('src.ingestion.db.pg_pool.SimpleConnectionPool', return_value=mock_pool):
        conn = get_connection()
        assert conn.closed == 0
        release_connection(conn)
        mock_pool.getconn.assert_called_once()
        mock_pool.putconn.assert_called_once_with(mock_conn)


def test_execute_many_inserts_rows():
    """Test that execute_many inserts multiple rows correctly using mock."""
    # Create a mock connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_conn.cursor.return_value.__exit__.return_value = None

    # Call execute_many with mocked execute_batch
    sql = "INSERT INTO _test_em VALUES (%s)"
    rows = [("a",), ("b",)]

    with patch('psycopg2.extras.execute_batch') as mock_execute_batch:
        execute_many(mock_conn, sql, rows)

        # Verify that execute_batch was called exactly once with correct arguments
        mock_execute_batch.assert_called_once_with(mock_cursor, sql, rows, page_size=1000)

    # Verify commit was called
    mock_conn.commit.assert_called_once()

    # Verify the cursor context manager was used correctly
    mock_conn.cursor.assert_called_once()
