import streamlit as st
from llama_cpp import Llama

# Path to your Meta-Llama-3.1-8B GGUF model
MODEL_PATH = "/Users/elenas/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Load the model
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  # Adjust context size as needed

# Streamlit app
st.title("Chat with Elena's bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask me anything..."):
    # Display user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user's message in session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response from the LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm(f"User: {user_input}\nAssistant:", max_tokens=200, stop=["User:", "\n"])[
                "choices"][0]["text"].strip()
            st.markdown(response)

    # Save assistant's response in session state
    st.session_state.messages.append({"role": "assistant", "content": response})