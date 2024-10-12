import re
import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_backend.rag_engine import RAGEngine as engine

GENERATION_COST = 0.00
conversation_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_backend", "conversation.txt")

def store_conversation(user, ai, cost):
    os.makedirs(os.path.dirname(conversation_file), exist_ok=True)
    with open(conversation_file, "a") as f:
        f.write(f"User: {user}\n")
        f.write(f"Assistant: {ai}\n\n")
        f.write(f"Total Cost: ${cost}\n\n")

# Initialize the RAG engine
@st.cache_resource
def _get_rag_engine():
    return engine()

def update_total_cost():
    st.session_state.total_cost_placeholder.caption(f"Total cost: ${st.session_state.total_cost:.2f}")

def setup_page():
    # Set up the Streamlit page
    st.set_page_config(page_title="ChatHuberman", page_icon="🤖")
    st.title("ChatHuberman")

    rag_engine = _get_rag_engine()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialise total conversation cost
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.00

    # Add checkbox for storing conversation logs
    if "store_logs" not in st.session_state:
        st.session_state.store_logs = False
    
    st.sidebar.checkbox("Enable conversation logging", key="store_logs")
    
    # Create a placeholder for the total cost in the sidebar
    if "total_cost_placeholder" not in st.session_state:
        st.session_state.total_cost_placeholder = st.sidebar.empty()
    
    # Update the total cost display
    update_total_cost()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                st.caption(f"Tokens: {message['metadata']['total_tokens']} | Cost: {message['metadata']['total_cost']}")

    # React to user input
    if prompt := st.chat_input("Enter the question you want answered using the Huberman Lab podcast"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response from RAG engine
        context = rag_engine.retrieve_relevant_documents(prompt)
        chat_history = "\n".join([m["content"] for m in st.session_state.messages])
        raw_response, generation_token_usage = rag_engine.chain(user_input=prompt, context=context, chat_history=chat_history, few_shot=True)
        response = re.sub(r'<thinking>.*?</thinking>', '', raw_response, flags=re.DOTALL)

        # Calculate costs
        input_tokens = generation_token_usage["input_tokens"]
        output_tokens = generation_token_usage["output_tokens"]
        total_tokens = generation_token_usage["total_tokens"]
        input_cost, output_cost, total_cost = rag_engine.calculate_generation_cost(input_tokens, output_tokens)
        st.session_state.total_cost += total_cost

        # Update the total cost display
        update_total_cost()

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(f"Tokens: {total_tokens} | Cost: ${total_cost:.6f}")

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "metadata": {
                "total_tokens": total_tokens,
                "total_cost": f"${total_cost:.6f}"
            }
        })

        # Store conversation logs if the checkbox is checked
        if st.session_state.store_logs:
            store_conversation(prompt, response, total_cost)

        # Manage chat history size
        if len(st.session_state.messages) > 10:
            st.session_state.messages = st.session_state.messages[2:]

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.total_cost = 0.00
        update_total_cost()
        st.rerun()

def main():
    setup_page()

if __name__ == "__main__":
    main()
