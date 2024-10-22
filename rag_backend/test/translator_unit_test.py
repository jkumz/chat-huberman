import os
import sys
from langchain_core.documents.base import Document
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag_backend.query_translator import QueryTranslator as qt

nested_sample_docs = [
    [Document(page_content="Content 1", metadata={"chunk_index": 0, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
     Document(page_content="Content 2", metadata={"chunk_index": 1, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"})],
    [Document(page_content="Content 2", metadata={"chunk_index": 1, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
     Document(page_content="Content 3", metadata={"chunk_index": 2, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"})],
    [
     Document(page_content="Content 3", metadata={"chunk_index": 2, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
     Document(page_content="Content 3", metadata={"chunk_index": 2, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"})],
]
nested_unique_sample_docs = [
    [
        Document(page_content="Content 1", metadata = {"chunk_index": 0, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
        Document(page_content="Content 2", metadata = {"chunk_index": 1, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
        Document(page_content="Content 3", metadata = {"chunk_index": 2, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
    ],
    [
        Document(page_content="Content 1", metadata = {"chunk_index": 0, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
        Document(page_content="Content 3", metadata = {"chunk_index": 2, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
    ]
]
unique_sample = [
    Document(page_content="Content 1", metadata = {"chunk_index": 0, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
    Document(page_content="Content 2", metadata = {"chunk_index": 1, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
    Document(page_content="Content 3", metadata = {"chunk_index": 2, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}),
]

# For local testing use env file, otherwise use environment variables in GitHub Actions
if not os.environ.get("OPENAI_API_KEY"):
    load_dotenv(find_dotenv(filename=".rag_engine.env"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
translator = qt(OPENAI_API_KEY)


def test_get_unique_union_when_not_unique():
    uniq_docs = translator.get_unique_union(nested_sample_docs)
    assert len(uniq_docs) == len(unique_sample), "Unique docs should be the same length as unique_sample"
    # Check that each document from unique_sample is in uniq_docs
    for doc in unique_sample:
        assert any(translator.serialise_doc(d) == translator.serialise_doc(doc) for d in uniq_docs), "Each document from unique_sample should be in uniq_docs"


def test_get_unique_union_when_unique():
    uniq_docs = translator.get_unique_union(nested_unique_sample_docs)
    assert len(uniq_docs) == len(unique_sample), "Unique docs should be the same length as unique_sample"
    # Check that each document from unique_sample is in uniq_docs
    for doc in unique_sample:
        assert any(translator.serialise_doc(d) == translator.serialise_doc(doc) for d in uniq_docs), "Each document from unique_sample should be in uniq_docs"

def test_serialise_doc():
    sample_doc = Document(page_content="Content 1", metadata = {"chunk_index": 0, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"})
    doc = translator.serialise_doc(sample_doc)
    assert doc == (0, 0, "123", "Huberman Lab", "https://www.youtube.com/watch?v=123", "Content 1"), "Serialised doc should be a tuple of the document's metadata and page content"

def test_deserialise_doc():
    sample_serialised_doc = (0, 0, "123", "Huberman Lab", "https://www.youtube.com/watch?v=123", "Content 1")
    doc = translator.deserialise_doc(sample_serialised_doc)
    assert doc.page_content == "Content 1", "Deserialised doc should have the same page content"
    assert doc.metadata == {"chunk_index": 0, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}, "Deserialised doc should have the same metadata"

def test_rrf():
    reranked_docs = translator.reciprocal_rank_fusion(nested_sample_docs)
    assert len(reranked_docs) == len(unique_sample), "Reranked docs should remove duplicates"
    for doc in unique_sample:
        assert any(translator.serialise_doc(d) == translator.serialise_doc(doc) for d in reranked_docs), "Each document from unique_sample should be in reranked_docs"

    assert reranked_docs[0].page_content == "Content 3" and reranked_docs[0].metadata == {"chunk_index": 2, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}
    assert reranked_docs[1].page_content == "Content 2" and reranked_docs[1].metadata == {"chunk_index": 1, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}
    assert reranked_docs[2].page_content == "Content 1" and reranked_docs[2].metadata == {"chunk_index": 0, "split_index": 0, "video_id": "123", "video_title": "Huberman Lab", "video_url": "https://www.youtube.com/watch?v=123"}


    