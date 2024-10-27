import asyncio
import streamlit as st
import aiohttp
import sys
import os
from dotenv import load_dotenv, find_dotenv
from api_key_validator import validate_api_keys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(find_dotenv(".env"))

conversation_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_backend", "conversation.txt")

def _store_conversation(user, ai, cost):
    """Stores the conversation in a file for logging + debugging purposes"""
    os.makedirs(os.path.dirname(conversation_file), exist_ok=True)
    with open(conversation_file, "a") as f:
        f.write(f"User: {user}\n")
        f.write(f"Assistant: {ai}\n\n")
        f.write(f"Total Cost: ${cost:.2f}\n\n")

async def _get_answer(prompt, chat_history):
    """
    Makes API call to RAG engine microservice

    Parameters:
    - prompt: The user's input query/question which needs answering
    - chat_history: The chat history to be used for context

    Returns:
    - The response from the RAG engine microservice
    """
    try:
        url = os.getenv("ENGINE_URL")
        headers = {
            "Content-Type": "application/json",
            "X-OpenAI-API-Key": st.session_state.openai_api_key,
            "X-Anthropic-API-Key": st.session_state.anthropic_api_key
        }
        json_data = {
            "user_input": prompt,
            "history": chat_history,
            "few_shot": True,
            "format_response": True
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=json_data, headers=headers) as response:
                return await response.json()
    except aiohttp.ClientError as e:
        if isinstance(e, aiohttp.ClientError):
            status_code = e.response.status_code
            if status_code == 400:
                st.error("Bad request: The server couldn't understand the request.")
            elif status_code == 401:
                st.error("Unauthorized: Authentication failed. Please check your API keys.")
            elif status_code == 403:
                st.error("Forbidden: You don't have permission to access this resource.")
            elif status_code == 404:
                st.error("Not Found: The requested resource could not be found.")
            elif status_code == 429:
                st.error("Too Many Requests: You've exceeded the rate limit. Please try again later.")
            elif status_code >= 500:
                st.error("Server Error: Something went wrong on the server side. Please try again later.")
            else:
                st.error(f"HTTP Error {status_code}: An unexpected error occurred.")
        elif isinstance(e, aiohttp.ClientError):
            st.error("Connection Error: Failed to connect to the server. Please check your internet connection.")
        elif isinstance(e, aiohttp.ClientTimeout):
            st.error("Timeout Error: The request timed out. Please try again later.")
        else:
            st.error(f"An unexpected error occurred: {str(e)}")
        return None
    except Exception as unknown_error:
        st.error(f"An unexpected error occurred: {str(unknown_error)}")
        return None

def _update_total_cost(cost=0.00):
    st.session_state.total_cost = cost
    if "total_cost_placeholder" in st.session_state:
        st.session_state.total_cost_placeholder.caption(f"Total cost: ${st.session_state.total_cost:.2f}")

def _centre_spinners():
    """
    Centres the loading spinners in Streamlit
    """
    st.markdown("""
    <style>
    div.stSpinner > div {
        text-align:center;
        align-items: center;
        justify-content: center;
    }
    </style>""", unsafe_allow_html=True)

def _initialise_session_state():
    """
    Initialises the session state variables
    """

    # Initialise API keys
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = ""

    # Initialize session state variables
    if 'api_keys_accepted' not in st.session_state:
        st.session_state.api_keys_accepted = False
    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0.00
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "block_processing" not in st.session_state:
        st.session_state.block_processing = False

    # Create a placeholder for the total cost in the sidebar
    if "total_cost_placeholder" not in st.session_state:
        st.session_state.total_cost_placeholder = st.sidebar.empty()
        st.session_state.total_cost_placeholder.caption(f"Total cost: ${st.session_state.total_cost:.2f}")

    # Function to validate and update API keys
    def validate_and_update_keys():
        if st.session_state.openai_api_key and st.session_state.anthropic_api_key:
            if validate_api_keys(st.session_state.openai_api_key, st.session_state.anthropic_api_key):
                st.session_state.api_keys_accepted = True
                st.sidebar.success("API keys validated successfully!")
                st.rerun()
            else:
                st.sidebar.error("Invalid API keys. Please check and try again.")
        elif st.session_state.openai_api_key or st.session_state.anthropic_api_key:
            st.sidebar.warning("Please enter both API keys in the sidebar to continue.")

    # Check if API keys have been accepted
    if not st.session_state.api_keys_accepted:
        st.session_state.openai_api_key = st.sidebar.text_input(
            label="Enter your OpenAI API key",
            type="password",
            key="openai_api_key_input",
            value=st.session_state.openai_api_key
        )
        st.session_state.anthropic_api_key = st.sidebar.text_input(
            label="Enter your Anthropic API key",
            type="password",
            key="anthropic_api_key_input",
            value=st.session_state.anthropic_api_key
        )
        if st.sidebar.button("Validate API Keys"):
            validate_and_update_keys()
    else:
        # Hide the text input boxes when API keys are accepted
        st.sidebar.markdown("API Keys Accepted")

    return st.session_state.api_keys_accepted

def _add_answer_to_history(answer, total_cost):
    st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "total_cost": f"${total_cost:.6f}"
                    }
                })
    
def _display_prompt_and_add_to_history(prompt):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

def _display_answer(answer, total_cost):
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"Cumulative Message Cost: ${total_cost:.6f}")

def _resize_history():
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[2:]

def _display_previous_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if "metadata" in message:
            st.caption(f"Cumulative Message Cost: {message['metadata']['total_cost']}")

async def setup_page():
    """
    Sets up the Streamlit page control flow
    """
    st.set_page_config(page_title="ChatHuberman", page_icon="🤖")
    st.title("ChatHuberman")
    st.caption("DISCLAIMER: This has no association with the Huberman Lab podcast or Andrew Huberman. This was a fun learning project.")

    api_keys_accepted = _initialise_session_state()

    if api_keys_accepted:
        _centre_spinners()

        st.sidebar.checkbox("Enable conversation logging", key="store_logs")

        _update_total_cost(st.session_state.total_cost)
        _display_previous_messages()

        # React to user input
        if not st.session_state.processing:
            user_input = st.chat_input("Enter the question you want answered using the Huberman Lab podcast", key="user_input")
            if user_input:
                st.session_state.processing = True
                st.session_state.current_input = user_input
                st.rerun()
        else:
            st.chat_input("Processing...", disabled=True)

        if st.session_state.processing:
            prompt = st.session_state.current_input
            _display_prompt_and_add_to_history(prompt)

            try:
                # Get response from RAG engine
                with st.spinner("Gathering relevant information and synthesising response..."):
                    chat_history = "\n".join([m["content"] for m in st.session_state.messages])
                    response = await _get_answer(prompt, chat_history)
                    answer = response["answer"]
                    generation_cost = response["generation_cost"]
                    retrieval_cost = response["retrieval_cost"]
                    translation_cost = response["translation_cost"]

                # Calculate costs
                total_cost = st.session_state.total_cost + generation_cost + retrieval_cost + translation_cost
                _update_total_cost(total_cost)
                _display_answer(answer, total_cost)
                _add_answer_to_history(answer, total_cost)

                # Store conversation logs if the checkbox is checked
                if st.session_state.store_logs:
                    _store_conversation(prompt, response, total_cost)

                # Manage chat history size
                _resize_history()

            except Exception as e:
                st.error(f"Error getting answer: {e}")

            finally:
                st.session_state.processing = False
                st.session_state.current_input = None
                st.rerun()

    def clear_chat_history_callback():
        st.session_state.messages = []
        st.session_state.total_cost = 0.00
        _update_total_cost()
        st.rerun()

    # Add a button to clear chat history
    st.button("Clear Chat History", key="clear_chat_history", on_click=clear_chat_history_callback)

def main():
    asyncio.run(setup_page())

if __name__ == "__main__":
    main()
