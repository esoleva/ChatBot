import streamlit as st
import os
from llama_cpp import Llama

# Path to your Meta-Llama model
MODEL_PATH = "/Users/elenas/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Path to your local folder with text files
FOLDER_PATH = "/Users/elenas/AWS courses"

# Load the model only once
if "llm" not in st.session_state:
    st.session_state.llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)

llm = st.session_state.llm  # Use the stored model

# Function to load text from files in the folder
def load_text_from_folder(folder_path):
    if not os.path.exists(folder_path):  # Ensure folder exists
        return "Error: Folder not found."

    text_data = ""
    for filename in os.listdir(folder_path):
        if filename == "presentations_txt.txt":  # Only process this specific file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text_data += f"\n\n--- {filename} ---\n\n" + file.read()
    
    return text_data if text_data else "No text files found."

# Load knowledge base
knowledge_base = load_text_from_folder(FOLDER_PATH)


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
                max_tokens=256,  # Lower max tokens for faster responses
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