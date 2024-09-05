import pytest
from unittest.mock import patch, MagicMock
from components.transcript_indexer import Indexer

@pytest.fixture
def mock_external_services():
    # Mock all external services and dependencies used by the Indexer
    with patch('components.transcript_indexer.AI21SemanticTextSplitter') as mock_splitter, \
         patch('components.transcript_indexer.OpenAIEmbeddings') as mock_embeddings, \
         patch('components.transcript_indexer.Pinecone') as mock_pinecone, \
         patch('components.transcript_indexer.YoutubeDL') as mock_ytdl, \
         patch('components.transcript_indexer.VideoProcessingTracker') as mock_tracker:
        
        # Set up mock behaviors
        mock_splitter.return_value.split_text.return_value = ["Split 1", "Split 2"]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_pinecone.return_value.Index.return_value = MagicMock()
        mock_ytdl.return_value.__enter__.return_value.extract_info.return_value = {"title": "Test Video"}
        
        yield {
            'splitter': mock_splitter,
            'embeddings': mock_embeddings,
            'pinecone': mock_pinecone,
            'ytdl': mock_ytdl,
            'tracker': mock_tracker
        }

@pytest.fixture
def indexer(mock_external_services):
    # Create an Indexer instance with mock API keys
    return Indexer(
        ai21_api_key="mock_ai21_key",
        openai_api_key="mock_openai_key",
        pinecone_api_key="mock_pinecone_key"
    )

def test_tracker_indexer_interaction(indexer, mock_external_services):
    # Test the interaction between Indexer and VideoProcessingTracker during successful processing
    url = "https://www.youtube.com/watch?v=test_video"
    chunks = ["This is a long enough chunk to be processed"]
    video_id = "test_video"

    indexer.process_and_index_chunks(url, chunks, video_id)

    # Verify that the tracker methods are called correctly
    mock_external_services['tracker'].return_value.start_processing.assert_called_once_with(video_id)
    mock_external_services['tracker'].return_value.complete_processing.assert_called_once_with(video_id)
    mock_external_services['tracker'].return_value.fail_processing.assert_not_called()

def test_indexer_tracker_interaction_on_error(indexer, mock_external_services):
    # Test the interaction between Indexer and VideoProcessingTracker when an error occurs
    url = "https://www.youtube.com/watch?v=test_video"
    chunks = ["This is a long enough chunk to be processed"]
    video_id = "test_video"

    # Simulate an error in semantic splitting
    mock_external_services['splitter'].return_value.split_text.side_effect = Exception("Splitting error")

    # Expect the method to raise an exception
    # trunk-ignore(ruff/B017)
    with pytest.raises(Exception):
        indexer.process_and_index_chunks(url, chunks, video_id)

    # Verify that the tracker methods are called correctly in case of an error
    mock_external_services['tracker'].return_value.start_processing.assert_called_once_with(video_id)
    mock_external_services['tracker'].return_value.complete_processing.assert_not_called()
    mock_external_services['tracker'].return_value.fail_processing.assert_called_once_with(video_id)