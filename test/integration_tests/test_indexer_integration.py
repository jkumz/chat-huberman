"""We don't want to test Pinecone, AI21 Semantic Splitting, or OpenAI embedding. These are external
services and they are maintained. We can presume they are working."""

from unittest.mock import MagicMock, patch

import pytest

from components.transcript_indexer import Indexer


@pytest.fixture
def mock_external_services():
    with patch(
        "components.transcript_indexer.AI21SemanticTextSplitter"
    ) as mock_splitter, patch(
        "components.transcript_indexer.OpenAIEmbeddings"
    ) as mock_embeddings, patch(
        "components.transcript_indexer.Pinecone"
    ) as mock_pinecone:
        # Setup mock behaviors
        mock_splitter.return_value.split_text.return_value = ["Split 1", "Split 2"]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_pinecone.return_value.Index.return_value = MagicMock()

        yield {
            "splitter": mock_splitter,
            "embeddings": mock_embeddings,
            "pinecone": mock_pinecone
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
