import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths
MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace sentence transformer
DATA_PATH = "/Users/elenas/AWScourses/presentations.txt"

# Load the Sentence Transformer model
model = SentenceTransformer(MODEL_PATH)

# Function to clean and preprocess the text
def clean_text(text):
    text = text.lower().replace("\n", " ").replace("\r", "")
    return " ".join(text.split())

# Function to load or create the local documents
def load_or_create_documents():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return []

    # Try reading the file
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as file:
            text = file.read().strip()
            print(f"File content (first 500 characters): {text[:500]}...")  # Debugging
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    if not text:
        print("No content found in the file.")
        return []

    # Clean and split the text
    text = clean_text(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    print(f"Created {len(chunks)} chunks.")
    if chunks:
        print(f"Preview of first chunk: {chunks[0][:200]}...")

    return [Document(page_content=chunk) for chunk in chunks]

# Load the documents
documents = load_or_create_documents()

# Function to retrieve relevant information
def retrieve_info(query):
    print(f"User query: {query}")  # Debugging
    relevant_docs = []

    for doc in documents:
        print(f"Checking chunk (first 100 chars): {doc.page_content[:100]}...")
        if query.lower() in doc.page_content.lower():
            relevant_docs.append(doc.page_content)

    if not relevant_docs:
        print("No relevant documents found.")

    return "\n".join(relevant_docs) if relevant_docs else None

# Streamlit UI
st.title("Chat with Elena's bot (RAG-enabled)")

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
                response = f"🤖 **Source: AI model**\n\nSorry, I couldn't find an answer from the local file."

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
