"""
Integration Tests for the Scraper class

These tests verify the correct functioning of the Scraper class as a whole,
including the interaction between its various methods. While some external
dependencies are mocked, these tests focus on ensuring that the different
components of the Scraper class work together correctly to scrape and
preprocess YouTube transcripts under various scenarios.
"""

from unittest.mock import Mock, patch

import pytest

from components.transcript_scraper import Scraper


# Mock the scrapetube module to avoid actual API calls during testing
@pytest.fixture
def mock_scrapetube():
    # Use patch to mock the scrapetube module within the Scraper class
    with patch("components.transcript_scraper.scrapetube") as mock:
        # Set up the mock to return a list of 5 mock video IDs when get_channel is called
        mock.get_channel.return_value = [
            {"videoId": f"mockvideo{i}"} for i in range(1, 6)
        ]
        yield mock  # Provide the mock to the test, then clean up after


# Mock the YoutubeLoader to avoid actual YouTube API calls during testing
@pytest.fixture
def mock_youtube_loader():
    # Use patch to mock the YoutubeLoader within the Scraper class
    with patch("components.transcript_scraper.YoutubeLoader") as mock:
        # Create a mock transcript with repeated content
        mock_transcript = Mock()
        mock_transcript.page_content = "Mock transcript content " * 5000

        # Create a mock loader instance
        mock_instance = Mock()
        mock_instance.load.return_value = [mock_transcript]

        # Set up the mock so that from_youtube_url returns our mock instance
        mock.from_youtube_url.return_value = mock_instance
        return mock


# Create a Scraper instance with mocked dependencies for testing
@pytest.fixture
def scraper(mock_scrapetube, mock_youtube_loader):
    return Scraper("mockuser", youtube_loader=mock_youtube_loader)


# Test that scrape_and_preprocess returns a list of the correct length
# Set a 5-second timeout for this test
def test_scrape_and_preprocess_returns_list(scraper):
    result = scraper.scrape_and_preprocess(count=2)
    assert isinstance(result, list)
    assert len(result) == 2, f"Expected 2 items, but got {len(result)}"


# Test that scrape_and_preprocess returns dictionaries with the correct structure


def test_scrape_and_preprocess_returns_valid_dictionaries(scraper):
    result = scraper.scrape_and_preprocess(count=2)
    for item in result:
        assert isinstance(item, dict)
        assert "url" in item
        assert "chunks" in item
        assert isinstance(item["chunks"], list)
        assert "www.youtube.com/watch?v" in item["url"]


# Test that the chunks returned by scrape_and_preprocess are of the correct size


def test_scrape_and_preprocess_validates_chunk_sizes(scraper):
    result = scraper.scrape_and_preprocess(count=2)
    for item in result:
        chunks = item["chunks"]
        assert all(len(chunk) <= 99999 for chunk in chunks)
        if len(chunks) > 1:
            assert len(chunks[0]) == 99999


# Test that scrape_and_preprocess processes all videos when no count is specified


def test_scrape_and_preprocess_with_no_count(scraper):
    result = scraper.scrape_and_preprocess()
    assert len(result) == 5, f"Expected 5 items (all URLs), but got {len(result)}"


# Test that scrape_and_preprocess handles exceptions from the YouTube loader gracefully


def test_scrape_and_preprocess_handles_youtube_loader_exception(
    scraper, mock_youtube_loader
):
    # Simulate an exception being raised by the YouTube loader
    mock_youtube_loader.from_youtube_url.side_effect = Exception("YouTube API error")
    result = scraper.scrape_and_preprocess(count=1)
    assert len(result) == 0, "Expected empty list when YouTube loader fails"


# Edge case: Test behavior when scrapetube returns an empty list of videos


def test_scrape_and_preprocess_with_no_videos(scraper, mock_scrapetube):
    # Override the mock to return an empty list
    mock_scrapetube.get_channel.return_value = []
    result = scraper.scrape_and_preprocess()
    assert len(result) == 0, "Expected empty list when no videos are available"


# Edge case: Test behavior when YouTube returns an empty transcript


def test_scrape_and_preprocess_with_empty_transcript(scraper, mock_youtube_loader):
    # Override the mock to return an empty transcript
    mock_youtube_loader.from_youtube_url.return_value.load.return_value = [
        Mock(page_content="")
    ]
    result = scraper.scrape_and_preprocess(count=1)
    assert len(result) == 1, "Expected one item even with empty transcript"
    assert len(result[0]["chunks"]) == 0, "Expected no chunks for empty transcript"


# Edge case: Test behavior when YouTube returns a very short transcript


def test_scrape_and_preprocess_with_short_transcript(scraper, mock_youtube_loader):
    # Override the mock to return a very short transcript
    mock_youtube_loader.from_youtube_url.return_value.load.return_value = [
        Mock(page_content="Short content")
    ]
    result = scraper.scrape_and_preprocess(count=1)
    assert len(result) == 1, "Expected one item for short transcript"
    assert (
        len(result[0]["chunks"]) == 1
    ), f"Expected one chunk for short transcript, got {len(result[0]['chunks'])}"
    assert len(result[0]["chunks"][0]) == len(
        "Short content"
    ), "Chunk should contain entire short transcript"


# Edge case: Test behavior when YouTube returns a transcript exactly at the chunk size limit


def test_scrape_and_preprocess_with_exact_chunk_size_transcript(
    scraper, mock_youtube_loader
):
    # Create a transcript that's exactly 99999 characters long
    exact_content = "a" * 99999
    mock_youtube_loader.from_youtube_url.return_value.load.return_value = [
        Mock(page_content=exact_content)
    ]
    result = scraper.scrape_and_preprocess(count=1)

    assert len(result) == 1, "Expected one item for exact chunk size transcript"
    assert (
        len(result[0]["chunks"]) == 1
    ), f"Expected one chunk for exact chunk size transcript, got {len(result[0]['chunks'])}"
    assert (
        len(result[0]["chunks"][0]) == 99999
    ), f"Expected chunk size of 99999, got {len(result[0]['chunks'][0])}"


# Test behaviour when YouTube returns a transcript 150000 characters long


def test_scrape_and_preprocess_with_multiple_chunks(scraper, mock_youtube_loader):
    # Create a transcript that's 150000 characters long
    multi_chunk_content = "a" * 150000
    mock_youtube_loader.from_youtube_url.return_value.load.return_value = [
        Mock(page_content=multi_chunk_content)
    ]
    result = scraper.scrape_and_preprocess(count=1)

    assert len(result) == 1, "Expected one item for multi-chunk transcript"
    assert (
        len(result[0]["chunks"]) == 2
    ), f"Expected two chunks for multi-chunk transcript, got {len(result[0]['chunks'])}"
    assert (
        len(result[0]["chunks"][0]) == 99999
    ), f"Expected first chunk size of 99999, got {len(result[0]['chunks'][0])}"
    assert (
        len(result[0]["chunks"][1]) == 50001
    ), f"Expected second chunk size of 50001, got {len(result[0]['chunks'][1])}"


# Test behaviour when YouTube returns a transcript 50000 characters long


def test_scrape_and_preprocess_with_small_transcript(scraper, mock_youtube_loader):
    # Create a transcript that's smaller than the chunk size
    small_content = "a" * 50000
    mock_youtube_loader.from_youtube_url.return_value.load.return_value = [
        Mock(page_content=small_content)
    ]
    result = scraper.scrape_and_preprocess(count=1)

    assert len(result) == 1, "Expected one item for small transcript"
    assert (
        len(result[0]["chunks"]) == 1
    ), f"Expected one chunk for small transcript, got {len(result[0]['chunks'])}"
    assert (
        len(result[0]["chunks"][0]) == 50000
    ), f"Expected chunk size of 50000, got {len(result[0]['chunks'][0])}"


# Test behaviour when YouTube returns a transcript 2500000 characters long


def test_scrape_and_preprocess_with_large_transcript(scraper, mock_youtube_loader):
    # Create a transcript that's 250000 characters long
    large_content = "a" * 250000
    mock_youtube_loader.from_youtube_url.return_value.load.return_value = [
        Mock(page_content=large_content)
    ]
    result = scraper.scrape_and_preprocess(count=1)

    assert len(result) == 1, "Expected one item for large transcript"
    assert (
        len(result[0]["chunks"]) == 3
    ), f"Expected three chunks for large transcript, got {len(result[0]['chunks'])}"
    assert (
        len(result[0]["chunks"][0]) == 99999
    ), f"Expected first chunk size of 99999, got {len(result[0]['chunks'][0])}"
    assert (
        len(result[0]["chunks"][1]) == 99999
    ), f"Expected second chunk size of 99999, got {len(result[0]['chunks'][1])}"
    assert (
        len(result[0]["chunks"][2]) == 50002
    ), f"Expected third chunk size of 50002, got {len(result[0]['chunks'][2])}"


# Edge case: Test behavior when scrapetube raises an exception


def test_scrape_and_preprocess_with_scrapetube_exception(scraper, mock_scrapetube):
    # Simulate an exception in scrapetube
    mock_scrapetube.get_channel.side_effect = Exception("Scrapetube API error")
    result = scraper.scrape_and_preprocess()
    assert len(result) == 0, "Expected empty list when scrapetube fails"


# Edge case: Test behavior when YouTube returns transcripts in different languages


def test_scrape_and_preprocess_with_different_languages(scraper, mock_youtube_loader):
    def side_effect(url, language):
        if language == "en-US":
            raise Exception("No English (US) transcript")
        return Mock(load=lambda: [Mock(page_content="Transcript in another language")])

    mock_youtube_loader.from_youtube_url.side_effect = side_effect
    result = scraper.scrape_and_preprocess(count=1)
    assert len(result) == 1, "Expected one item when falling back to another language"
    assert "Transcript in another language" in result[0]["chunks"][0]
