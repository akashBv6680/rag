import streamlit as st
import os
import sys
import tempfile
import uuid
import json
import requests
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import shutil

# This block ensures a compatible sqlite3 version is used
# by replacing the default system version with pysqlite3.
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    st.error("pysqlite3 is not installed. Please add 'pysqlite3-binary' to your requirements.txt.")
    st.stop()

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"
# API key is provided by the user
# ⚠️ REMINDER: DO NOT HARDCODE YOUR API KEY IN A PUBLIC REPOSITORY. USE SECRETS OR ENVIRONMENT VARIABLES.
# This key is a placeholder and should be replaced with your actual key.
TOGETHER_API_KEY = "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Use Streamlit's cache to initialize dependencies once
@st.cache_resource
def initialize_dependencies():
    """
    Initializes and returns the ChromaDB client and SentenceTransformer model.
    Using @st.cache_resource ensures this runs only once.
    """
    try:
        db_path = tempfile.mkdtemp()
        db_client = chromadb.PersistentClient(path=db_path)
        # Explicitly load the model to the CPU to avoid PyTorch errors
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        return db_client, model
    except Exception as e:
        st.error(f"An error occurred during dependency initialization: {e}.")
        st.stop()

def get_db_path():
    """Returns a unique temporary path for the ChromaDB directory."""
    if "db_path" not in st.session_state:
        st.session_state.db_path = tempfile.mkdtemp()
    return st.session_state.db_path

def call_together_api(prompt, max_retries=5):
    """
    Calls the Together AI API with exponential backoff for retries.
    """
    if TOGETHER_API_KEY == "YOUR_TOGETHER_API_KEY_HERE":
        st.error("Please replace 'YOUR_TOGETHER_API_KEY_HERE' with your actual Together AI API key.")
        return {"error": "API key not set"}

    retry_delay = 1
    for i in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOGETHER_API_KEY}"
            }
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [{"role": "system", "content": "You are a helpful assistant."},
                             {"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}")
            if e.response.status_code == 429:
                st.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            elif e.response.status_code == 401:
                st.error("Invalid API Key. Please check your Together AI API key.")
                return {"error": "401 Unauthorized"}
            else:
                st.error(f"Failed to call API after {i+1} retries: {e}")
                return {"error": str(e)}
        except Exception as e:
            st.error(f"An error occurred during the API call: {e}")
            return {"error": str(e)}

def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    try:
        if 'db_client' in st.session_state and COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def get_collection():
    """Retrieves or creates the ChromaDB collection."""
    return st.session_state.db_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=len, is_separator_regex=False
    )
    return splitter.split_text(text_data)

def is_valid_github_raw_url(url):
    """Checks if a URL is a valid GitHub raw file URL."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w./-]+\.(txt|md)"
    return re.match(pattern, url) is not None

def process_and_store_documents(documents):
    """
    Processes a list of text documents, generates embeddings, and
    stores them in ChromaDB.
    """
    collection = get_collection()
    model = st.session_state.model
    embeddings = model.encode(documents).tolist()
    document_ids = [str(uuid.uuid4()) for _ in documents]
    
    collection.add(documents=documents, embeddings=embeddings, ids=document_ids)
    st.toast("Documents processed and stored successfully!", icon="✅")

def retrieve_documents(query, n_results=5):
    """
    Retrieves the most relevant documents from ChromaDB based on a query.
    """
    collection = get_collection()
    model = st.session_state.model
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results['documents'][0]

def rag_pipeline(query):
    """
    Executes the full RAG pipeline: retrieval and response generation.
    """
    collection = get_collection()
    if collection.count() == 0:
        return "Please upload a document or provide a GitHub raw URL before asking questions."

    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs)
    prompt = (f"Using the following information, answer the user's question. If the information is not present, "
              f"state that you cannot answer. \n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:")
    
    response_json = call_together_api(prompt)

    if 'error' in response_json:
        return "An error occurred while generating the response. Please try again."
    
    try:
        return response_json['choices'][0]['message']['content']
    except (KeyError, IndexError):
        st.error("Invalid API response format.")
        return "Failed to get a valid response from the model."

def display_chat_messages():
    """Displays all chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handles new user input, runs the RAG pipeline, and updates chat history."""
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_pipeline(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(layout="wide", page_title="RAG Chat Flow")
    
    # Initialize dependencies with a cached function
    if 'db_client' not in st.session_state or 'model' not in st.session_state:
        st.session_state.db_client, st.session_state.model = initialize_dependencies()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    st.title("RAG Chat Flow")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload text files (.txt)", type="txt", accept_multiple_files=True)
        github_url = st.text_input("Enter a GitHub raw `.txt` or `.md` URL:")

        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    for uploaded_file in uploaded_files:
                        file_contents = uploaded_file.read().decode("utf-8")
                        documents = split_documents(file_contents)
                        process_and_store_documents(documents)
                    st.success("All files processed and stored! You can now ask questions about them.")
        
        if github_url:
            if st.button("Process URL"):
                if is_valid_github_raw_url(github_url):
                    with st.spinner("Fetching and processing file from URL..."):
                        try:
                            response = requests.get(github_url)
                            response.raise_for_status()
                            file_contents = response.text
                            documents = split_documents(file_contents)
                            process_and_and_store_documents(documents)
                            st.success("File from URL processed! You can now chat about its contents.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error fetching URL: {e}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                else:
                    st.error("Please enter a valid GitHub raw URL (ending in .txt or .md).")
        
        st.markdown("---")
        st.header("Chat Controls")
        if st.button("Start New Chat"):
            st.session_state.messages = []
            clear_chroma_data()
            st.experimental_rerun()
    
    # Display chat messages
    display_chat_messages()
    
    # Handle user input
    handle_user_input()

if __name__ == "__main__":
    main_ui()
