import streamlit as st
import os
import sys
import tempfile
import uuid
import json
import requests
import time
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import shutil

# Import a compatible version of sqlite3 for chromadb
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"
# API key is provided by the user
TOGETHER_API_KEY = "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

def get_db_path():
    """Returns a unique temporary path for the ChromaDB directory."""
    # Use a unique temporary directory to avoid conflicts between reruns
    if "db_path" not in st.session_state:
        st.session_state.db_path = tempfile.mkdtemp()
    return st.session_state.db_path

def initialize_chroma_client():
    """Initializes and returns a ChromaDB client."""
    db_path = get_db_path()
    return chromadb.PersistentClient(path=db_path)

def get_sentence_transformer_model():
    """Initializes and returns the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def call_together_api(prompt, max_retries=5):
    """
    Calls the Together AI API with exponential backoff for retries.
    """
    retry_delay = 1
    for i in range(max_retries):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOGETHER_API_KEY}"
            }

            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }

            response = requests.post(TOGETHER_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for bad status codes
            
            return response.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}")
            if e.response.status_code == 429: # Too Many Requests
                st.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
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
    # This function is now part of the UI logic and will be triggered by a button
    try:
        if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
            st.session_state.db_client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Error clearing collection: {e}")

def get_collection():
    """
    Retrieves or creates the ChromaDB collection.
    """
    if 'db_collection' not in st.session_state:
        try:
            st.session_state.db_collection = st.session_state.db_client.get_collection(
                name=COLLECTION_NAME
            )
        except Exception:
            st.session_state.db_collection = st.session_state.db_client.get_or_create_collection(
                name=COLLECTION_NAME
            )
    return st.session_state.db_collection

def split_documents(text_data, chunk_size=500, chunk_overlap=100):
    """Splits a single string of text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text_data)

def is_valid_github_raw_url(url):
    """Checks if a URL is a valid GitHub raw file URL."""
    pattern = r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[^/]+/[\w-./]+\.(txt|md)"
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
    
    # Add documents to the ChromaDB collection
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=document_ids
    )

    st.toast("Documents processed and stored successfully!", icon="✅")

def retrieve_documents(query, n_results=5):
    """
    Retrieves the most relevant documents from ChromaDB based on a query.
    """
    collection = get_collection()
    model = st.session_state.model
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results['documents'][0]

def rag_pipeline(query):
    """
    Executes the full RAG pipeline: retrieval and response generation.
    """
    relevant_docs = retrieve_documents(query)
    
    # Create the grounded prompt for the LLM
    context = "\n".join(relevant_docs)
    prompt = f"Using the following information, answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Call the Together API to get a real-time response
    response_json = call_together_api(prompt)

    if 'error' in response_json:
        return "An error occurred while generating the response. Please try again."
    
    try:
        # Extract the text from the API response
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
    if prompt := st.chat_input("Ask about your family..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_pipeline(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Streamlit UI ---
def main_ui():
    """Sets up the main Streamlit UI for the RAG chatbot."""
    st.set_page_config(layout="wide")

    # Sidebar
    with st.sidebar:
        st.header("RAG Chat Flow")
        if st.button("New Chat"):
            st.session_state.messages = []
            clear_chroma_data()
            st.session_state.chat_history = {}
            st.session_state.current_chat_id = None
            st.session_state.current_date = datetime.now().strftime("%b %d, %I:%M %p")
            st.experimental_rerun()

        # Chat history display logic
        st.subheader("Chat History")
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            sorted_chat_ids = sorted(
                st.session_state.chat_history.keys(), 
                key=lambda x: st.session_state.chat_history[x]['date'], 
                reverse=True
            )
            for chat_id in sorted_chat_ids:
                chat_title = st.session_state.chat_history[chat_id]['title']
                date_str = st.session_state.chat_history[chat_id]['date'].strftime("%b %d, %I:%M %p")
                if st.button(f"**{chat_title}** - {date_str}", key=chat_id):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
                    st.experimental_rerun()

    # Main content area
    st.title("RAG Chat Flow")
    st.markdown("---")

    # Document upload/processing section
    with st.container():
        st.subheader("Add Context Documents")
        uploaded_file = st.file_uploader("Upload a text file (.txt)", type="txt")
        github_url = st.text_input("Enter a GitHub raw `.txt` or `.md` URL:")

        if uploaded_file:
            file_contents = uploaded_file.read().decode("utf-8")
            st.session_state.uploaded_file_content = file_contents
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    documents = split_documents(file_contents)
                    process_and_store_documents(documents)
                    st.success("File processed! You can now ask questions about its content.")

        if github_url and is_valid_github_raw_url(github_url):
            if st.button("Process URL"):
                with st.spinner("Fetching and processing file from URL..."):
                    try:
                        response = requests.get(github_url)
                        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
                        file_contents = response.text
                        documents = split_documents(file_contents)
                        process_and_store_documents(documents)
                        st.success("File from URL processed! You can now chat about its contents.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error fetching URL: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize db client and model directly in session state
    if 'db_client' not in st.session_state:
        db_path = get_db_path()
        st.session_state.db_client = chromadb.PersistentClient(path=db_path)
    if 'model' not in st.session_state:
        st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.chat_history[st.session_state.current_chat_id] = {
            'messages': st.session_state.messages,
            'title': "New Chat",
            'date': datetime.now()
        }

    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main_ui()
