import os
import sys
import pytest
import asyncio
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents.base import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag_backend.rag_engine import RAGEngine as engine

"""
We are only testing that the engine returns valid answer and context, the quality and correctness
of the answer and context is assessed in our evaluation class
"""

# For local testing use env file, otherwise use environment variables in GitHub Actions
if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    load_dotenv(find_dotenv(filename=".rag_engine.env"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# @pytest.fixture(scope="module")
# def event_loop():
#     loop = asyncio.get_event_loop()
#     yield loop
#     loop.close()

@pytest.fixture(scope="module")
def rag_engine():
    return engine(OPENAI_API_KEY, ANTHROPIC_API_KEY)

@pytest.mark.asyncio(scope="module")
async def test_get_answer_returns_dict(rag_engine):
    hippocampus_answer = await rag_engine.get_answer("What is the purpose of the hippocampus?", few_shot=True, history="", format_response=True)
    assert isinstance(hippocampus_answer, dict), f"get_answer should return a dictionary, instead got: {hippocampus_answer}"
    assert "answer" in hippocampus_answer, "Dictionary should have 'answer' key"
    assert "generation_cost" in hippocampus_answer, "Dictionary should have 'generation_cost' key"
    assert "retrieval_cost" in hippocampus_answer, "Dictionary should have 'retrieval_cost' key"
    assert "translation_cost" in hippocampus_answer, "Dictionary should have 'translation_cost' key"

@pytest.mark.asyncio(scope="module")
async def test_get_answer_contains_youtube_links(rag_engine):
    hippocampus_answer = await rag_engine.get_answer("What is the purpose of the hippocampus?", few_shot=True, history="", format_response=True)
    print(f"Answer: {hippocampus_answer}")  # Debug print
    assert "youtube.com" in hippocampus_answer["answer"], "Answer should contain links to YouTube videos used to answer the question"
    assert "<thinking>" not in hippocampus_answer["answer"] and "</thinking>" not in hippocampus_answer["answer"], "Answer should not contain <thinking> tags"

@pytest.mark.asyncio(scope="module")
async def test_get_answer_returns_non_empty_string(rag_engine):
    hippocampus_answer = await rag_engine.get_answer("What is the purpose of the hippocampus?", few_shot=True, history="", format_response=True)
    print(f"Answer: {hippocampus_answer}")  # Debug print
    assert len(hippocampus_answer["answer"]) > 0, "Answer should not be empty"
    assert isinstance(hippocampus_answer["answer"], str), "Answer should be a string"

@pytest.mark.asyncio(scope="module")
async def test_get_answer_with_context_returns_answer_with_thinking_tags(rag_engine):
    hippocampus_answer_with_context = await rag_engine.get_answer_with_context("What is the purpose of the hippocampus?", few_shot=True)
    print(f"Answer with context: {hippocampus_answer_with_context}")  # Debug print
    answer = hippocampus_answer_with_context["answer"]
    assert "<thinking>" in answer and "</thinking>" in answer, "Answer should contain <thinking> tags"

@pytest.mark.asyncio(scope="module")
async def test_get_answer_with_context_returns_non_empty_string_answer(rag_engine):
    hippocampus_answer_with_context = await rag_engine.get_answer_with_context("What is the purpose of the hippocampus?", few_shot=True)
    answer = hippocampus_answer_with_context["answer"]
    assert len(answer) > 0, "Answer should not be empty"
    assert isinstance(answer, str), "Answer should be a string"

@pytest.mark.asyncio(scope="module")
async def test_get_answer_with_context_returns_non_empty_context(rag_engine):
    hippocampus_answer_with_context = await rag_engine.get_answer_with_context("What is the purpose of the hippocampus?", few_shot=True)
    context = hippocampus_answer_with_context["context"]
    assert len(context) > 0, "Context should not be empty"
    assert isinstance(context, list), "Context should be a list"
    assert all(isinstance(doc, Document) for doc in context), "All items in context should be Document objects"

@pytest.mark.asyncio(scope="module")
async def test_get_answer_with_context_returns_list_of_documents(rag_engine):
    hippocampus_answer_with_context = await rag_engine.get_answer_with_context("What is the purpose of the hippocampus?", few_shot=True)
    context = hippocampus_answer_with_context["context"]
    assert isinstance(context, list), "Context should be a list"
    assert all(isinstance(doc, Document) for doc in context), "All documents should be LangChain Document objects"

@pytest.mark.asyncio(scope="module")
async def test_get_answer_with_context_returns_dict(rag_engine):
    hippocampus_answer_with_context = await rag_engine.get_answer_with_context("What is the purpose of the hippocampus?", few_shot=True)
    assert isinstance(hippocampus_answer_with_context, dict), "get_answer_with_context should return a dictionary"
    assert "answer" in hippocampus_answer_with_context and "context" in hippocampus_answer_with_context, "Dictionary should have 'answer' and 'context' keys"

@pytest.mark.asyncio(scope="module")
async def test_retrieve_relevant_documents_returns_non_empty_list(rag_engine):
    hippocampus_retrieve_relevant_documents = await rag_engine.retrieve_relevant_documents("What is the purpose of the hippocampus?")
    print(f"Retrieved documents: {hippocampus_retrieve_relevant_documents}")  # Debug print
    assert len(hippocampus_retrieve_relevant_documents) > 0, "Documents should not be empty"
    assert isinstance(hippocampus_retrieve_relevant_documents, list), "Documents should be a list"

@pytest.mark.asyncio(scope="module")
async def test_retrieve_relevant_documents_returns_list_of_documents(rag_engine):
    hippocampus_retrieve_relevant_documents = await rag_engine.retrieve_relevant_documents("What is the purpose of the hippocampus?")
    assert isinstance(hippocampus_retrieve_relevant_documents, list), "Documents should be a list"
    assert all(isinstance(doc, Document) for doc in hippocampus_retrieve_relevant_documents), "All documents should be LangChain Document objects"

@pytest.mark.asyncio(scope="module")
async def test_retrieve_relevant_documents_returns_documents_with_correct_metadata(rag_engine):
    hippocampus_retrieve_relevant_documents = await rag_engine.retrieve_relevant_documents("What is the purpose of the hippocampus?")
    assert all(len(d.metadata) == 5 for d in hippocampus_retrieve_relevant_documents), "All documents should have 5 metadata fields"
    assert all(d.metadata.get('video_id') for d in hippocampus_retrieve_relevant_documents), "All documents should have a video_id"
    assert all(d.metadata.get('video_title') for d in hippocampus_retrieve_relevant_documents), "All documents should have a video_title"
    assert all(d.metadata.get('video_url') for d in hippocampus_retrieve_relevant_documents), "All documents should have a video_url"
    assert all('chunk_index' in d.metadata and d.metadata['chunk_index'] is not None for d in hippocampus_retrieve_relevant_documents), "All documents should have a chunk_index"
    assert all(d.metadata.get('split_index') for d in hippocampus_retrieve_relevant_documents), "All documents should have a split_index"
    assert all(d.page_content for d in hippocampus_retrieve_relevant_documents), "All documents should have a page_content"
