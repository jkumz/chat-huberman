"""
The purpose of this class is to test the integration of the Streamlit app with the RAG
engine, as well as OpenAI, Anthropic, LangChain and Pinecone libraries via the Rag Engine
integration within it.

To comprehensively test the integration, we will use real API keys passed in as environment
variables and a real instance of the RAG engine using these keys.

To test the actual Streamlit app, we will use the Streamlit testing framework
(https://docs.streamlit.io/develop/api-reference/app-testing)
"""

from streamlit.testing.v1 import AppTest

import os
import sys
from dotenv import load_dotenv, find_dotenv

# Add the parent directory of 'main' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# For local testing use env file, otherwise use environment variables in GitHub Actions
if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    load_dotenv(find_dotenv(filename=".test.env"))

# Import methods to unit test - not integrated
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Helper function to simulate the app
def simulate_app():
    at = AppTest.from_file("main/app.py")
    return at

# Test that the session state is initialised correctly
def test_session_state_initialisation():
    app = simulate_app()
    app.run()

    assert app.session_state.messages == [], "Chat history should be empty"
    assert not app.session_state.api_keys_accepted, "API keys should not have been accepted"
    assert app.session_state.total_cost == 0.00, f"Total cost should be $0.00, real value: {app.session_state.total_cost}"
    assert "total_cost_placeholder" in app.session_state, "Total cost placeholder should exist"
    assert not app.session_state.processing, "Processing should not be active with no keys/prompt entered"
    assert not app.session_state.block_processing, "Processing blocker should not be active with no keys/prompt entered"

# Test that generation is not possible when no API keys are accepted
def test_user_input_blocked_when_no_api_keys_accepted():
    app = simulate_app()
    app.run()

    assert not app.session_state.api_keys_accepted, "API keys should not have been accepted"
    assert not app.session_state.processing, "Processing should not be active"
    assert not app.session_state.block_processing, "Processing blocker should not be active"
    assert "rag_engine" not in app.session_state, "RAG engine should not have been initialised"
    assert app.session_state.messages == [], "Chat history should be empty"

# Test that the user can enter valid API keys
def test_user_keys_accepted():
    app = simulate_app()
    app.run()

    # Simulate entering valid API key
    app.text_input(key="openai_api_key").set_value(OPENAI_API_KEY)
    app.text_input(key="anthropic_api_key").set_value(ANTHROPIC_API_KEY)

    # Force rerun to ensure the session state is updated
    app.run(timeout=30) # High timeout as we are waiting on generated resp from Claude, not always fast

    assert app.session_state.api_keys_accepted, "API keys should have been accepted"
    assert 'rag_engine' in app.session_state, "RAG engine should have been initialized"

# Test that the user cannot enter invalid API keys
def test_invalid_api_keys():
    app = simulate_app()
    app.run()

    # Simulate entering invalid API keys
    app.text_input(key="openai_api_key").set_value("invalid_openai_key")
    app.text_input(key="anthropic_api_key").set_value("invalid_anthropic_key")
    app.run()

    assert not app.session_state.api_keys_accepted, "API keys should not have been accepted"

# Test that the user cannot enter incomplete API keys
def test_incomplete_api_keys():
    app = simulate_app()
    app.run()

    # Simulate entering only one Open AI key
    app.text_input(key="openai_api_key").set_value(OPENAI_API_KEY)
    app.run()

    assert not app.session_state.api_keys_accepted, "API keys should not have been accepted"

    # Simulate entering only one Anthropic key
    app.text_input(key="openai_api_key").set_value("")
    app.text_input(key="anthropic_api_key").set_value(ANTHROPIC_API_KEY)
    app.run()

    assert not app.session_state.api_keys_accepted, "API keys should not have been accepted"

# Test that when API keys are accepted, user can enter a prompt and get a response
def test_user_prompt_and_response():
    app = simulate_app()
    app.run()

    # Simulate entering valid API keys
    app.text_input(key="openai_api_key").set_value(OPENAI_API_KEY)
    app.text_input(key="anthropic_api_key").set_value(ANTHROPIC_API_KEY)
    app.run(timeout=10)

    # Simulate entering a prompt
    app.chat_input(key="user_input").set_value("What is the main function of the amygdala?").run(timeout=30)
    assert app.session_state.messages[0]["content"] == "What is the main function of the amygdala?", "First message should be the user prompt"
    assert len(app.session_state.messages) == 2, "The messages should contain two items: the user prompt and the response"
    assert app.session_state.total_cost > 0.00, "Total cost should be greater than $0.00"

    app.button(key="clear_chat_history").click().run()
    assert app.session_state.messages == [], f"Chat history should be empty, real length: {len(app.session_state.messages)}"
    assert app.session_state.total_cost == 0.00, "Total cost should be $0.00"
