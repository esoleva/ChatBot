import streamlit as st
from llama_cpp import Llama

# Path to your Meta-Llama-3.1-8B GGUF model
MODEL_PATH = "/Users/elenas/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Load the model with optimized parameters
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=8)  # Adjust as needed

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
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm(
                f"User: {user_input}\nAssistant:",
                max_tokens=512,  # Lower max tokens for faster responses
                stop=["User:", "\n\nUser:"],  # Better stop sequence
            )["choices"][0]["text"].strip()

            # Debugging: Print raw response
            print(f"Raw Response: {response}")

            # Format response for markdown
            formatted_response = "\n".join(line.strip() for line in response.split("\n"))
            formatted_response = formatted_response.replace("\n-", "\n\n-")
            formatted_response = formatted_response.replace("\n*", "\n\n*")

            st.markdown(formatted_response)

    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
