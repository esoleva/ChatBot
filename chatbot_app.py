import streamlit as st
import os
import concurrent.futures
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# Paths
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embedding model
DATA_PATH = "/Users/elenas/AWScourses/presentations.txt"
LLAMA_MODEL_PATH = "/Users/elenas/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Load models
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=4096, n_threads=8)  # Adjust as needed

# Function to clean and preprocess the text
def clean_text(text):
    text = text.lower().replace("\n", " ").replace("\r", "")
    return " ".join(text.split())

# Function to load or create the local documents
def load_or_create_documents():
    if not os.path.exists(DATA_PATH):
        return []

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as file:
            text = file.read().strip()
    except Exception:
        return []

    if not text:
        return []

    text = clean_text(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    return [Document(page_content=chunk) for chunk in chunks]

# Load the documents
documents = load_or_create_documents()

# Function to retrieve relevant information
def retrieve_info(query):
    relevant_docs = [
        doc.page_content for doc in documents if query.lower() in doc.page_content.lower()
    ]
    
    if relevant_docs:
        return "\n".join([f"• {doc}\n" for doc in relevant_docs])
    else:
        return None

# Function to get an AI-generated response as fallback using Llama with timeout
def get_llama_response(query, timeout=20):
    def llama_call():
        return llm(
            f"User: {query}\nAssistant:",
            max_tokens=512,  # Lower max tokens for faster responses
            stop=["User:", "\n\nUser:"],  # Stop sequences to prevent excessive generation
        )["choices"][0]["text"].strip()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(llama_call)
        try:
            return future.result(timeout=timeout)  # Set timeout
        except concurrent.futures.TimeoutError:
            return "Llama model timed out. Please try again."

# Streamlit UI
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

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_info = retrieve_info(user_input)

            if retrieved_info:
                response = f"📁 **Source: Local file**\n\n{retrieved_info}"
            else:
                llama_response = get_llama_response(user_input)
                response = f"🤖 **Source: Llama Model**\n\n{llama_response}"

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
