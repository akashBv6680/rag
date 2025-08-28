import streamlit as st
import os
import sys
import tempfile
import uuid
# Import a compatible version of sqlite3 for chromadb
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import shutil

# --- Constants and Configuration ---
COLLECTION_NAME = "rag_documents"

def get_db_path():
    """Returns the path to the ChromaDB directory."""
    return "./chroma_db"

def initialize_chroma():
    """Initializes and returns a ChromaDB client."""
    db_path = get_db_path()
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    return chromadb.PersistentClient(path=db_path)

def clear_chroma_data():
    """Clears all data from the ChromaDB collection."""
    if COLLECTION_NAME in [col.name for col in st.session_state.db_client.list_collections()]:
        st.session_state.db_client.delete_collection(name=COLLECTION_NAME)

def get_collection():
    """
    Retrieves or creates the ChromaDB collection.
    
    This function is now a core part of the app's state management,
    ensuring a single, persistent collection is used across runs.
    """
    if 'db_collection' not in st.session_state or st.session_state.db_collection is None:
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

def is_valid_github_url(url):
    """Checks if a URL is a valid GitHub repository URL."""
    pattern = r"https://github\.com/[\w-]+/[\w-]+"
    return re.match(pattern, url) is not None

def process_and_store_documents(documents):
    """
    Processes a list of text documents, generates embeddings, and
    stores them in ChromaDB.
    """
    model = st.session_state.model
    collection = get_collection()

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
    model = st.session_state.model
    collection = get_collection()
    
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
    
    # A simple, concatenated prompt for response generation
    context = "\n".join(relevant_docs)
    prompt = f"Using the following information, answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Call the LLM (this is a placeholder for your actual LLM call)
    # The actual LLM call would be an API call here.
    # For now, we'll return a placeholder response
    return "This is a placeholder response based on the retrieved context."

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
            # Sort chats by creation date descending
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
        github_url = st.text_input("Enter a GitHub repository URL to load `.md` or `.txt` files from:")

        if uploaded_file:
            # Handle uploaded file
            file_contents = uploaded_file.read().decode("utf-8")
            st.session_state.uploaded_file_content = file_contents
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    documents = split_documents(file_contents)
                    process_and_store_documents(documents)
                    st.success("File processed! You can now ask questions about its content.")

        if github_url and is_valid_github_url(github_url):
            if st.button("Process GitHub Repo"):
                with st.spinner("Cloning and processing repository..."):
                    temp_dir = tempfile.mkdtemp()
                    # Placeholder for Git cloning logic
                    # try:
                    #     from git import Repo
                    #     Repo.clone_from(github_url, temp_dir)
                    #     docs_text = ""
                    #     for root, _, files in os.walk(temp_dir):
                    #         for file in files:
                    #             if file.endswith((".md", ".txt")):
                    #                 file_path = os.path.join(root, file)
                    #                 with open(file_path, "r", encoding="utf-8") as f:
                    #                     docs_text += f.read() + "\n"
                    #     documents = split_documents(docs_text)
                    #     process_and_store_documents(documents)
                    #     st.success("Repository processed! You can now chat about its contents.")
                    # except Exception as e:
                    #     st.error(f"Error processing repository: {e}")
                    # finally:
                    #     shutil.rmtree(temp_dir)
                    pass

    # Initialize chat history and ChromaDB client in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'db_client' not in st.session_state:
        st.session_state.db_client = initialize_chroma()
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

