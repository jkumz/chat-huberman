"""We don't want to test Pinecone, AI21 Semantic Splitting, or OpenAI embedding. These are external
services and they are maintained. We can presume they are working."""

import os
from unittest.mock import MagicMock, patch

import pytest

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
    with patch(
        "components.transcript_indexer.AI21SemanticTextSplitter"
    ) as mock_splitter, patch(
        "components.transcript_indexer.OpenAIEmbeddings"
    ) as mock_embeddings, patch(
        "components.transcript_indexer.Pinecone"
    ) as mock_pinecone, patch(
        "components.transcript_indexer.VideoProcessingTracker"
    ) as mock_tracker:
        # Setup mock behaviors
        mock_splitter.return_value.split_text.return_value = ["Split 1", "Split 2"]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_pinecone.return_value.Index.return_value = MagicMock()
        mock_tracker.return_value = MagicMock()
        yield {
            "splitter": mock_splitter,
            "embeddings": mock_embeddings,
            "pinecone": mock_pinecone,
            "tracker": mock_tracker
        }


@pytest.fixture
def indexer(mock_external_services):
    return Indexer(
        ai21_api_key="mock_ai21_key",
        openai_api_key="mock_openai_key",
        pinecone_api_key="mock_pinecone_key",
    )


def test_process_and_index_chunks_flow(indexer, mock_external_services):
    url = "https://www.youtube.com/watch?v=test_video"
    chunks = [
        "This is chunk 1This is chunk 1This is chunk 1",
        "This is chunk 2This is chunk 2This is chunk 2",
    ]
    title = "Test Video"
    video_id = "test_video"

    indexer.process_and_index_chunks(url, chunks, video_id, title)

    # Verify that external services were called correctly
    mock_external_services["splitter"].return_value.split_text.assert_called()
    mock_external_services["embeddings"].return_value.embed_query.assert_called()

    # Verify that Pinecone upsert was called
    indexer.index.upsert.assert_called()

    # Verify the structure of data sent to Pinecone
    calls = indexer.index.upsert.call_args_list
    assert len(calls) > 0
    vectors = calls[0][1]["vectors"]
    assert len(vectors) > 0
    for vector in vectors:
        assert len(vector) == 3  # (id, embedding, metadata)
        assert isinstance(vector[0], str)  # id
        assert isinstance(vector[1], list)  # embedding
        assert isinstance(vector[2], dict)  # metadata
        assert "video_id" in vector[2]
        assert "video_url" in vector[2]
        assert "video_title" in vector[2]
        assert "text" in vector[2]


def test_batch_processing(indexer, mock_external_services):
    url = "https://www.youtube.com/watch?v=test_video"
    chunks = [
        "ChunkChunkChunkChunkChunkChunk"
    ] * 150  # Create 150 chunks to test batch processing
    video_id = "test_video"
    title = "Test Video"
    indexer.process_and_index_chunks(url, chunks, video_id, title)

    # Verify that Pinecone upsert was called multiple times (due to batching)
    assert indexer.index.upsert.call_count > 1, "Batching upsert to Pinecone failed"

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
    # trunk-ignore(ruff/E712)
    # trunk-ignore(bandit/B101)
    assert delete_call.kwargs['delete_all'] == False
