# trunk-ignore-all(isort)
import pytest
from unittest.mock import patch, MagicMock
from components.video_processing_tracker import VideoProcessingTracker
from psycopg2 import sql
from freezegun import freeze_time
from datetime import datetime


@pytest.fixture
def mock_db_connection():
    # Create a mock PostgreSQL connection
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    yield conn, cursor
    conn.close()


@pytest.fixture
def tracker(mock_db_connection):
    # Create a VideoProcessingTracker instance with a mock database connection
    conn, _ = mock_db_connection
    with patch('psycopg2.connect', return_value=conn):
        tracker = VideoProcessingTracker(db_url='mock://connection')
        tracker.create_table()  # Ensure this method is called
        yield tracker
        tracker.close()


def test_create_table(tracker, mock_db_connection):
    # Test if the video_processing_status table is created upon initialization
    _, cursor = mock_db_connection
    expected_sql = """
        CREATE TABLE IF NOT EXISTS video_processing_status
        (video_id TEXT PRIMARY KEY, 
         status TEXT, 
         start_time TIMESTAMP,
         end_time TIMESTAMP)
        """
    cursor.execute.assert_called_with(expected_sql)


@freeze_time("2023-01-01 12:00:00")
def test_start_processing(tracker, mock_db_connection):
    # Test if start_processing correctly inserts a new record with 'processing' status
    _, cursor = mock_db_connection
    video_id = "test_video"
    cursor.fetchone.return_value = None  # Simulate video not existing

    tracker.start_processing(video_id)

    cursor.execute.assert_any_call(
        "SELECT status FROM video_processing_status WHERE video_id = %s",
        (video_id,)
    )
    cursor.execute.assert_called_with(
        """
                INSERT INTO video_processing_status
                (video_id, status, start_time) VALUES (%s, %s, %s)
                ON CONFLICT (video_id) DO UPDATE
                SET status = EXCLUDED.status, start_time = EXCLUDED.start_time
                """,
        (video_id, "processing", datetime.now())
    )


@freeze_time("2023-01-01 12:00:00")
def test_complete_processing(tracker, mock_db_connection):
    # Test if complete_processing updates the status to 'completed' and sets the end_time
    _, cursor = mock_db_connection
    video_id = "test_video"

    tracker.complete_processing(video_id)

    cursor.execute.assert_called_with(
        """
            UPDATE video_processing_status
            SET status = 'completed', end_time = %s
            WHERE video_id = %s
            """,
        (datetime.now(), video_id)
    )


@freeze_time("2023-01-01 12:00:00")
def test_fail_processing(tracker, mock_db_connection):
    # Test if fail_processing updates the status to 'failed' and sets the end_time
    _, cursor = mock_db_connection
    video_id = "test_video"

    tracker.fail_processing(video_id)

    cursor.execute.assert_called_with(
        """
            UPDATE video_processing_status
            SET status = 'failed', end_time = %s
            WHERE video_id = %s
            """,
        (datetime.now(), video_id)
    )


def test_get_status(tracker, mock_db_connection):
    # Test if get_status returns the correct status for a video
    _, cursor = mock_db_connection
    video_id = "test_video"
    cursor.fetchone.return_value = ("processing",)

    status = tracker.get_status(video_id)
    assert status == "processing"

    cursor.execute.assert_called_with(
        "SELECT status FROM video_processing_status WHERE video_id = %s",
        (video_id,)
    )


def test_get_unprocessed_videos(tracker, mock_db_connection):
    # Test if get_unprocessed_videos returns videos that are not marked as completed
    _, cursor = mock_db_connection
    cursor.fetchall.return_value = [("video2",), ("video3",)]

    unprocessed = tracker.get_unprocessed_videos()
    assert set(unprocessed) == {"video2", "video3"}

    cursor.execute.assert_called_with(
        "SELECT video_id FROM video_processing_status WHERE status != 'completed'"
    )


def test_check_if_video_exists_and_completed(tracker, mock_db_connection):
    # Test if check_if_video_exists_and_completed returns True if the video exists and is completed
    _, cursor = mock_db_connection
    video_id = "test_video"
    cursor.fetchone.return_value = (video_id, "completed")

    result = tracker.check_if_video_exists_and_completed(video_id)
    assert result is True

    cursor.execute.assert_called_with(
        'SELECT video_id, status FROM video_processing_status WHERE video_id = %s',
        (video_id,)
    )


@patch('os.environ.get')
def test_init_with_environment_variable(mock_env_get):
    # Test initialization with the DATABASE_URL environment variable
    mock_env_get.return_value = "postgresql://user:password@host:port/dbname"

    with patch('psycopg2.connect') as mock_connect:
        tracker = VideoProcessingTracker()
        mock_connect.assert_called_once_with("postgresql://user:password@host:port/dbname")


def test_init_with_custom_url():
    # Test initialization with a custom database URL
    custom_url = "postgresql://custom:url@host:port/dbname"

    with patch('psycopg2.connect') as mock_connect:
        tracker = VideoProcessingTracker(db_url=custom_url)
        mock_connect.assert_called_once_with(custom_url)


def test_init_without_url():
    # Test initialization without a URL (should raise ValueError)
    with patch('os.environ.get', return_value=None):
        with pytest.raises(ValueError):
            VideoProcessingTracker()
