# trunk-ignore-all(isort)
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from components.video_processing_tracker import VideoProcessingTracker


@pytest.fixture
def mock_db_connection():
    # Create an in-memory SQLite database for testing
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def tracker(mock_db_connection):
    # Create a VideoProcessingTracker instance with a mock database connection
    with patch(
        "components.video_processing_tracker.sqlite3.connect",
        return_value=mock_db_connection,
    ):
        tracker = VideoProcessingTracker(db_path=":memory:")
        yield tracker
        tracker.close()


def test_create_table(tracker, mock_db_connection):
    # Test if the video_processing_status table is created upon initialization
    cursor = mock_db_connection.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='video_processing_status'"
    )
    assert cursor.fetchone() is not None


def test_start_processing(tracker, mock_db_connection):
    # Test if start_processing correctly inserts a new record with 'processing' status
    video_id = "test_video"
    tracker.start_processing(video_id)

    cursor = mock_db_connection.cursor()
    cursor.execute(
        "SELECT status, start_time FROM video_processing_status WHERE video_id = ?",
        (video_id,),
    )
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == "processing"
    assert result[1] is not None  # Check if start_time is set


def test_complete_processing(tracker, mock_db_connection):
    # Test if complete_processing updates the status to 'completed' and sets the end_time
    video_id = "test_video"
    tracker.start_processing(video_id)
    tracker.complete_processing(video_id)

    cursor = mock_db_connection.cursor()
    cursor.execute(
        "SELECT status, end_time FROM video_processing_status WHERE video_id = ?",
        (video_id,),
    )
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == "completed"
    assert result[1] is not None  # Check if end_time is set


def test_fail_processing(tracker, mock_db_connection):
    # Test if fail_processing updates the status to 'failed' and sets the end_time
    video_id = "test_video"
    tracker.start_processing(video_id)
    tracker.fail_processing(video_id)

    cursor = mock_db_connection.cursor()
    cursor.execute(
        "SELECT status, end_time FROM video_processing_status WHERE video_id = ?",
        (video_id,),
    )
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == "failed"
    assert result[1] is not None  # Check if end_time is set


def test_get_status(tracker, mock_db_connection):
    # Test if get_status returns the correct status for a video
    video_id = "test_video"
    tracker.start_processing(video_id)

    status = tracker.get_status(video_id)
    assert status == "processing"

    tracker.complete_processing(video_id)
    status = tracker.get_status(video_id)
    assert status == "completed"


def test_get_unprocessed_videos(tracker, mock_db_connection):
    # Test if get_unprocessed_videos returns videos that are not marked as completed

    # Start processing for three videos
    tracker.start_processing("video1")
    tracker.start_processing("video2")
    tracker.start_processing("video3")

    # Complete processing for video1
    tracker.complete_processing("video1")

    # Fail processing for video2
    tracker.fail_processing("video2")

    # video3 is left in 'processing' state

    unprocessed = tracker.get_unprocessed_videos()
    # trunk-ignore(bandit/B101)
    assert set(unprocessed) == {"video2", "video3"}

    # Additional test to ensure completed videos are not returned
    tracker.complete_processing("video3")
    unprocessed = tracker.get_unprocessed_videos()
    assert set(unprocessed) == {"video2"}


@patch("os.path.exists")
@patch("os.path.join")
@patch("os.getcwd")
@patch("os.path.dirname")
@patch("sqlite3.connect")
def test_find_db_path(mock_connect, mock_dirname, mock_getcwd, mock_join, mock_exists):
    # Set up the mock directory structure
    mock_getcwd.return_value = "/fake/path/to/current/dir"

    def fake_dirname(path):
        parts = path.split("/")
        return "/".join(parts[:-1])

    mock_dirname.side_effect = fake_dirname

    mock_join.side_effect = lambda *args: "/".join(args)

    # Set up mock_exists to return True only for the correct path
    def fake_exists(path):
        return path == "/fake/path/persistence/video_processing.db"

    mock_exists.side_effect = fake_exists

    # Mock the sqlite3.connect method to prevent actual database creation
    mock_connect.return_value = MagicMock()

    tracker = VideoProcessingTracker()

    db_path = tracker.find_db_path()

    assert db_path == "/fake/path/persistence/video_processing.db"

    # Check that os.path.exists was called for each directory up to the correct one
    expected_checks = [
        "/fake/path/to/current/dir/persistence/video_processing.db",
        "/fake/path/to/current/persistence/video_processing.db",
        "/fake/path/to/persistence/video_processing.db",
        "/fake/path/persistence/video_processing.db",
    ]

    for check in expected_checks:
        mock_exists.assert_any_call(check)

    # Verify that sqlite3.connect was called with the correct path
    mock_connect.assert_called_once_with("/fake/path/persistence/video_processing.db")


@patch("components.video_processing_tracker.VideoProcessingTracker.find_db_path")
def test_init_with_default_path(mock_find_db_path):
    # Test initialization with the default (found) database path
    mock_find_db_path.return_value = "/fake/path/video_processing.db"

    with patch("sqlite3.connect") as mock_connect:
        # trunk-ignore(ruff/F841)
        tracker = VideoProcessingTracker()
        mock_connect.assert_called_once_with("/fake/path/video_processing.db")


def test_init_with_custom_path():
    # Test initialization with a custom database path
    custom_path = "/custom/path/video_processing.db"

    with patch("sqlite3.connect") as mock_connect:
        # trunk-ignore(ruff/F841)
        tracker = VideoProcessingTracker(db_path=custom_path)
        mock_connect.assert_called_once_with(custom_path)
