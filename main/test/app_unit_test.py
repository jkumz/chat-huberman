import os
import sys
from unittest.mock import patch
from app import _store_conversation

"""
For unit tests, we mock external dependencies; OpenAI; Anthropic
We will use the real Streamlit library, and the control flow of the app will be tested
However, we will mock the dependencies of the RAG Engine - Vector Store, Retriever, LLM, Output Parser, Query Translator

The RAG engine itself has it's own independent tests, so we don't need to test that.
"""

# Testing storing conversation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
test_conversation_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test", "test_convo.txt")

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