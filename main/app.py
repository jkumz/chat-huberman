import streamlit as st
import sys
import os
import openai
import anthropic

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_backend.rag_engine import RAGEngine as engine

conversation_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_backend", "conversation.txt")

def _check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

def _check_anthropic_api_key(api_key):
    client = anthropic.Client(api_key=api_key)
    try:
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1,
            messages=[
                {"role": "user", "content": "Test"}
                ]
            )
    except anthropic.AuthenticationError:
        return False
    else:
        return True

def _validate_api_keys(openai_key, anthropic_key):
    try:
        # Validate OpenAI API key
        openai_valid = _check_openai_api_key(openai_key)
        if not openai_valid:
            return False
        # Validate Anthropic API key
        anthropic_valid = _check_anthropic_api_key(anthropic_key)
        if not anthropic_valid:
            return False

        return True
    except Exception as e:
        st.error(f"API key validation failed: {str(e)}")
        return False

def _store_conversation(user, ai, cost):
    os.makedirs(os.path.dirname(conversation_file), exist_ok=True)
    with open(conversation_file, "a") as f:
        f.write(f"User: {user}\n")
        f.write(f"Assistant: {ai}\n\n")
        f.write(f"Total Cost: ${cost}\n\n")

# Initialize the RAG engine
@st.cache_resource
def _get_rag_engine(openai_api_key, anthropic_api_key):
    return engine(openai_api_key=openai_api_key, anthropic_api_key=anthropic_api_key)

def _update_total_cost(cost=0.00):
    st.session_state.total_cost = cost
    if "total_cost_placeholder" in st.session_state:
        st.session_state.total_cost_placeholder.caption(f"Total cost: ${st.session_state.total_cost:.2f}")

def _centre_spinners():
    st.markdown("""
    <style>
    div.stSpinner > div {
        text-align:center;
        align-items: center;
        justify-content: center;
    }
    </style>""", unsafe_allow_html=True)

def _initialise_session_state():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add API key inputs to the sidebar
    if 'api_keys_accepted' not in st.session_state:
        st.session_state.api_keys_accepted = False

    # Initialise total conversation cost
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.00

    # Create a placeholder for the total cost in the sidebar
    if "total_cost_placeholder" not in st.session_state:
        st.session_state.total_cost_placeholder = st.sidebar.empty()

    # Initialize processing state if it doesn't exist
    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "block_processing" not in st.session_state:
        st.session_state.block_processing = False

    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.00
        st.session_state.total_cost_placeholder.caption(f"Total cost: ${st.session_state.total_cost:.2f}")



    # Check if API keys have been accepted
    if not st.session_state.api_keys_accepted:
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
        anthropic_api_key = st.sidebar.text_input("Enter your Anthropic API key", type="password")

        if openai_api_key and anthropic_api_key:
            if _validate_api_keys(openai_api_key, anthropic_api_key):
                st.session_state.api_keys_accepted = True
                st.session_state.openai_api_key = openai_api_key
                st.session_state.anthropic_api_key = anthropic_api_key
                st.success("API keys validated successfully!")
            else:
                st.error("Invalid API keys. Please check and try again.")
        else:
            st.warning("Please enter both API keys in the sidebar to continue.")
            return
    else:
        # Hide the text input boxes when API keys are accepted
        st.sidebar.markdown("API Keys Accepted")

def setup_page():
    # Set up the Streamlit page
    st.set_page_config(page_title="ChatHuberman", page_icon="🤖")
    st.title("ChatHuberman")
    st.caption("DISCLAIMER: This has no association with the Huberman Lab podcast or Andrew Huberman. This was a fun learning project.")

    _initialise_session_state()

    # Initialize the RAG engine with API keys
    if st.session_state.api_keys_accepted:
        if 'rag_engine' not in st.session_state:
            st.session_state.rag_engine = _get_rag_engine(st.session_state.openai_api_key, st.session_state.anthropic_api_key)

        # Format spinners to center
        _centre_spinners()

        st.sidebar.checkbox("Enable conversation logging", key="store_logs")

        # Update the total cost display
        _update_total_cost(st.session_state.total_cost)

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "metadata" in message:
                    st.caption(f"Cumulative Message Cost: {message['metadata']['total_cost']}")

        # React to user input
        if not st.session_state.processing:
            user_input = st.chat_input("Enter the question you want answered using the Huberman Lab podcast")
            if user_input:
                st.session_state.processing = True
                st.session_state.current_input = user_input
                st.rerun()
        else:
            st.chat_input("Processing...", disabled=True)

        if st.session_state.processing:
            prompt = st.session_state.current_input

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                # Get response from RAG engine
                with st.spinner("Gathering relevant information and synthesising response..."):
                    chat_history = "\n".join([m["content"] for m in st.session_state.messages])
                    response = st.session_state.rag_engine.get_answer(user_input=prompt, few_shot=True, history=chat_history, format_response=True)

                # Calculate costs
                total_cost = st.session_state.rag_engine.get_generation_cost() + st.session_state.rag_engine.get_retrieval_cost() + st.session_state.rag_engine.get_translation_cost()
                # Update the total cost display
                _update_total_cost(total_cost)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.caption(f"Cumulative Message Cost: ${total_cost:.6f}")

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": {
                        "total_cost": f"${total_cost:.6f}"
                    }
                })

                # Store conversation logs if the checkbox is checked
                if st.session_state.store_logs:
                    _store_conversation(prompt, response, total_cost)

                # Manage chat history size
                with st.spinner("Updating agent memory..."):
                    if len(st.session_state.messages) > 10:
                        st.session_state.messages = st.session_state.messages[2:]

            finally:
                st.session_state.processing = False
                st.session_state.current_input = None
                st.rerun()

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.total_cost = 0.00
        _update_total_cost()
        st.rerun()

def main():
    setup_page()

if __name__ == "__main__":
    main()
