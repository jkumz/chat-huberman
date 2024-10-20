import os
import sys
from unittest.mock import patch
from app import _store_conversation, _get_rag_engine

"""
For unit tests, we mock external dependencies; OpenAI; Anthropic
We will use the real Streamlit library, and the control flow of the app will be tested
However, we will mock the dependencies of the RAG Engine - Vector Store, Retriever, LLM, Output Parser, Query Translator

The RAG engine itself has it's own independent tests, so we don't need to test that.
"""

# Testing storing conversation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
test_conversation_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main", "test", "test_convo.txt")

def test_store_conversation():
    # Mock the conversation file to isolate the test from the file system
    with patch('app.conversation_file', test_conversation_file):
        # Clear the file before writing
        open(test_conversation_file, 'w').close()
        
        _store_conversation("Mock User Prompt", "Mock AI Response", 0.00)
        assert os.path.exists(test_conversation_file)
        with open(test_conversation_file, "r") as f:
            assert f.read() == "User: Mock User Prompt\nAssistant: Mock AI Response\n\nTotal Cost: $0.00\n\n"
    
    # Clean up after the test
    if os.path.exists(test_conversation_file):
        os.remove(test_conversation_file)

# Mock the embedding model, vector store, retriever and LLM, query translator
@patch('rag_backend.rag_engine.OpenAIEmbeddings')
@patch('rag_backend.rag_engine.PineconeVectorStore')
@patch('rag_backend.rag_engine.ChatAnthropic')
@patch('rag_backend.rag_engine.QueryTranslator')
def test_get_rag_engine(mock_query_translator, mock_chat_anthropic, mock_pinecone_vector_store, mock_openai_embeddings):
    # Set up mock return values
    mock_openai_embeddings.return_value = "mock_embedding_model"
    mock_pinecone_vector_store.return_value.as_retriever.return_value = "mock_retriever"
    mock_chat_anthropic.return_value = "mock_llm"
    mock_query_translator.return_value = "mock_query_translator"

    # Call the function under test
    rag_engine = _get_rag_engine("mock_openai_key", "mock_anthropic_key")

    # Assert that the mocks were called with the correct arguments
    mock_openai_embeddings.assert_called_once_with(model="text-embedding-3-large", api_key="mock_openai_key")
    mock_chat_anthropic.assert_called_once()
    mock_query_translator.assert_called_once_with(openai_api_key="mock_openai_key")

    # Assert that the RAG engine was created with the mocked components
    assert rag_engine.embedding_model == "mock_embedding_model"
    assert rag_engine.retriever == "mock_retriever"
    assert rag_engine.llm == "mock_llm"
    assert rag_engine.query_translator == "mock_query_translator"
    assert rag_engine.generation_cost == 0.00
    assert rag_engine.retrieval_cost == 0.00
    assert rag_engine.translation_cost == 0.00