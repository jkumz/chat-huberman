import os
import pytest
from unittest.mock import patch, MagicMock
from components.transcript_indexer import Indexer

@pytest.fixture
def mock_environment():
    with patch.dict(os.environ, {
        "INDEX_SOURCE_TAG": "test_source_tag",
        "INDEX_HOST": "test_host",
        "INDEX_NAME": "test_index"
    }):
        yield

@pytest.fixture
def mock_external_services(mock_environment):
    # Mock all external services and dependencies used by the Indexer
    with patch('components.transcript_indexer.AI21SemanticTextSplitter') as mock_splitter, \
         patch('components.transcript_indexer.OpenAIEmbeddings') as mock_embeddings, \
         patch('components.transcript_indexer.Pinecone') as mock_pinecone, \
         patch('components.transcript_indexer.VideoProcessingTracker') as mock_tracker:
        
        # Set up mock behaviors
        mock_splitter.return_value.split_text.return_value = ["Split 1", "Split 2"]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_pinecone.return_value.Index.return_value = MagicMock()
        mock_tracker.return_value = MagicMock()
        
        yield {
            'splitter': mock_splitter,
            'embeddings': mock_embeddings,
            'pinecone': mock_pinecone,
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
    title = "Test Video"
    indexer.process_and_index_chunks(url, chunks, video_id, title)

    # Verify that the tracker methods are called correctly
    mock_external_services['tracker'].return_value.start_processing.assert_called_once_with(video_id)
    mock_external_services['tracker'].return_value.complete_processing.assert_called_once_with(video_id)
    mock_external_services['tracker'].return_value.fail_processing.assert_called_once_with(video_id)

def test_delete_existing_video_on_failure(indexer, mock_external_services):
    url = "https://www.youtube.com/watch?v=test_video"
    chunks = ["ChunkChunkChunkChunkChunkChunk"] * 150
    video_id = "test_video"
    title = "Test Video"

    # Mock the upsert method to raise an exception
    indexer.index.upsert.side_effect = Exception("Simulated failure")

    # Process should raise an exception
    with pytest.raises(Exception, match="Simulated failure"):
        indexer.process_and_index_chunks(url, chunks, video_id, title)

    # Verify that the tracker methods are called correctly
    mock_external_services['tracker'].return_value.start_processing.assert_called_once_with(video_id)
    mock_external_services['tracker'].return_value.fail_processing.assert_called_once_with(video_id)
    mock_external_services['tracker'].return_value.complete_processing.assert_not_called()

    # Verify that delete was called with the correct IDs
    indexer.index.delete.assert_called_once()
    delete_call = indexer.index.delete.call_args

    expected_vector_ids = [f"{video_id}_chunk{i}_split{j}" 
                           for i in range(len(chunks)) 
                           for j in range(100)]
    
    assert delete_call.kwargs['ids'] == expected_vector_ids
    assert delete_call.kwargs['delete_all'] == False
