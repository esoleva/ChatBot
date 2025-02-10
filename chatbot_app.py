import streamlit as st
from llama_cpp import Llama
import os

# Path to your Meta-Llama model
MODEL_PATH = "/Users/elenas/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Path to your local folder with text files
FOLDER_PATH = "/Users/elenas/AWS courses"

# Load the model
llm = Llama(model_path=MODEL_PATH, n_ctx=2096, n_threads=8)

# Function to load and preprocess text from files in the folder
def load_and_preprocess_text(folder_path, max_files=3, max_lines=100):
    if not os.path.exists(folder_path):  # Ensure folder exists
        return {}

    processed_text = {}
    file_count = 0
    
    # Process up to 'max_files' text files
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and file_count < max_files:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()[:max_lines]  # Only take the first N lines for faster load
                processed_text[filename] = "".join(lines)
                file_count += 1
    
    return processed_text

# Preprocess knowledge base once (stored in session state)
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = load_and_preprocess_text(FOLDER_PATH)

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

    # Search in the preprocessed knowledge base (simple keyword search)
    found_answer = None
    for filename, content in st.session_state.knowledge_base.items():
        if user_input.lower() in content.lower():  # Check if user input matches any part of the file content
            found_answer = content[:500]  # Show a portion of the found content
            source = f"from the file: {filename}"
            break

    if found_answer:
        # If content is found in the local knowledge base, return it
        response = f"Answer from {source}:\n{found_answer}"
    else:
        # If no match, use the model for a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm(f"User: {user_input}\nAssistant:", max_tokens=150)["choices"][0]["text"].strip()
                source = "general model knowledge"
    
    # Display the response with source information
    st.markdown(f"**Source**: {source}")
    st.markdown(response)

    # Save assistant's response in session state
    st.session_state.messages.append({"role": "assistant", "content": response})
