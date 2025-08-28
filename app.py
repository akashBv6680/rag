import streamlit as st
import requests
import os
import json
import uuid
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import shutil
from git import Repo

# Page configuration
st.set_page_config(
    page_title="RAG Chat Flow ✘",
    page_icon="✘",
    initial_sidebar_state="expanded"
)

# Initialize dark mode state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Define personality questions - reduced to general ones
PERSONALITY_QUESTIONS = [
    "What is [name]'s personality like?",
    "What does [name] do for work?",
    "What are [name]'s hobbies?",
    "What makes [name] special?",
    "Tell me about [name]"
]

# Enhanced CSS styling with dark mode support
def get_css_styles():
    """Returns CSS styles based on the dark_mode session state."""
    if st.session_state.dark_mode:
        return """
<style>
    /* Dark Mode Styles */
    .stApp {
        background: #0e1117;
        color: #fafafa;
    }
    
    .main .block-container {
        max-width: 900px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar dark mode */
    .css-1d391kg {
        background-color: #1e1e1e !important;
    }
    
    .css-1cypcdb {
        background-color: #1e1e1e !important;
    }
    
    /* Chat messages dark mode */
    .stChatMessage {
        background-color: #262730 !important;
        border: 1px solid #404040 !important;
    }
    
    /* Input fields dark mode */
    .stTextInput > div > div > input {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: #404040 !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: #404040 !important;
    }
    
    .model-id {
        color: #4ade80;
        font-family: monospace;
    }
    
    .model-attribution {
        color: #4ade80;
        font-size: 0.8em;
        font-style: italic;
    }
    
    .rag-attribution {
        color: #a78bfa;
        font-size: 0.8em;
        font-style: italic;
        background: #1f2937;
        padding: 8px;
        border-radius: 4px;
        border-left: 3px solid #a78bfa;
        margin-top: 8px;
    }
    
    /* Dark mode toggle button */
    .dark-mode-toggle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        margin: 4px 0;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9em;
        width: 100%;
        text-align: center;
    }
    
    .dark-mode-toggle:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* NEW CHAT BUTTON - Black background for dark mode */
    .stButton > button[kind="primary"] {
        background-color: #1f2937 !important;
        border-color: #374151 !important;
        color: #fafafa !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #374151 !important;
        border-color: #4b5563 !important;
        color: #fafafa !important;
    }
    
    /* Regular buttons dark mode */
    .stButton > button {
        background-color: #374151 !important;
        border-color: #4b5563 !important;
        color: #fafafa !important;
    }
    
    .stButton > button:hover {
        background-color: #4b5563 !important;
        border-color: #6b7280 !important;
        color: #fafafa !important;
    }
    
    /* Personality Questions Styling Dark Mode */
    .personality-question {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.85em;
        width: 100%;
        text-align: left;
    }
    
    .personality-question:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    .personality-section {
        background: #1f2937;
        color: #e5e7eb;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #4f46e5;
        margin: 10px 0;
    }
    
    /* Chat history styling dark mode */
    .chat-history-item {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        border: 1px solid #374151;
        background: #1f2937;
        color: #e5e7eb;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .chat-history-item:hover {
        background: #374151;
        border-color: #4ade80;
    }
    
    .document-status {
        background: #1e3a8a;
        color: #bfdbfe;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 10px 0;
    }
    
    .github-status {
        background: #581c87;
        color: #e9d5ff;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #a78bfa;
        margin: 10px 0;
    }
    
    .rag-stats {
        background: #581c87;
        color: #e9d5ff;
        padding: 8px;
        border-radius: 6px;
        font-size: 0.85em;
    }
    
    /* Expander dark mode */
    .streamlit-expanderHeader {
        background-color: #1f2937 !important;
        color: #fafafa !important;
    }
    
    .streamlit-expanderContent {
        background-color: #111827 !important;
        color: #fafafa !important;
    }
    
    /* Checkbox dark mode */
    .stCheckbox {
        color: #fafafa !important;
    }
    
    /* Select box dark mode */
    .stSelectbox > div > div {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    
    /* File uploader dark mode */
    .stFileUploader {
        background-color: #1f2937 !important;
        border-color: #374151 !important;
    }
    
    /* Progress bar dark mode */
    .stProgress .st-bo {
        background-color: #374151 !important;
    }
    
    /* Success/Error/Warning messages dark mode */
    .stSuccess {
        background-color: #064e3b !important;
        color: #6ee7b7 !important;
    }
    
    .stError {
        background-color: #7f1d1d !important;
        color: #fca5a5 !important;
    }
    
    .stWarning {
        background-color: #78350f !important;
        color: #fcd34d !important;
    }
    
    .stInfo {
        background-color: #1e3a8a !important;
        color: #93c5fd !important;
    }
    
    /* Caption text dark mode */
    .caption {
        color: #9ca3af !important;
    }
    
    /* Divider dark mode */
    hr {
        border-color: #374151 !important;
    }
</style>
"""
    else:
        return """
<style>
    /* Light Mode Styles */
    .stApp {
        background: white;
        color: #000000;
    }
    
    .main .block-container {
        max-width: 900px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .model-id {
        color: #28a745;
        font-family: monospace;
    }
    
    .model-attribution {
        color: #28a745;
        font-size: 0.8em;
        font-style: italic;
    }
    
    .rag-attribution {
        color: #6f42c1;
        font-size: 0.8em;
        font-style: italic;
        background: #f8f9fa;
        padding: 8px;
        border-radius: 4px;
        border-left: 3px solid #6f42c1;
        margin-top: 8px;
    }
    
    /* Light mode toggle button */
    .dark-mode-toggle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        margin: 4px 0;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9em;
        width: 100%;
        text-align: center;
    }
    
    .dark-mode-toggle:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* NEW CHAT BUTTON - Black background */
    .stButton > button[kind="primary"] {
        background-color: #000000 !important;
        border-color: #000000 !important;
        color: #ffffff !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #333333 !important;
        border-color: #333333 !important;
        color: #ffffff !important;
    }
    
    /* Personality Questions Styling */
    .personality-question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.85em;
        width: 100%;
        text-align: left;
    }
    
    .personality-question:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .personality-section {
        background: #f8f9ff;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    /* Chat history styling */
    .chat-history-item {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background: #f8f9fa;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .chat-history-item:hover {
        background: #e9ecef;
        border-color: #28a745;
    }
    
    .document-status {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    
    .github-status {
        background: #f3e5f5;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #6f42c1;
        margin: 10px 0;
    }
    
    .rag-stats {
        background: #f3e5f5;
        padding: 8px;
        border-radius: 6px;
        font-size: 0.85em;
        color: #4a148c;
    }
</style>
"""

# Apply CSS styles
st.markdown(get_css_styles(), unsafe_allow_html=True)

# File paths
HISTORY_FILE = "rag_chat_history.json"
SESSIONS_FILE = "rag_chat_sessions.json"
USERS_FILE = "online_users.json"


# ================= GITHUB INTEGRATION =================
def clone_github_repo():
    """Clone or update GitHub repository with documents"""
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        st.error("🔑 GITHUB_TOKEN not found in environment variables")
        return False
    
    try:
        repo_url = f"https://{github_token}@github.com/Umer-K/family-profiles.git"
        repo_dir = "family_profiles"
        
        # Clean up existing directory if it exists
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        
        # Clone the repository
        with st.spinner("🔄 Cloning private repository..."):
            Repo.clone_from(repo_url, repo_dir)
            
        # Copy txt files to documents folder
        documents_dir = "documents"
        os.makedirs(documents_dir, exist_ok=True)
        
        # Clear existing documents
        for file in os.listdir(documents_dir):
            if file.endswith('.txt'):
                os.remove(os.path.join(documents_dir, file))
        
        # Copy new txt files from repo
        txt_files_found = 0
        for root, dirs, files in os.walk(repo_dir):
            for file in files:
                if file.endswith('.txt'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(documents_dir, file)
                    shutil.copy2(src_path, dst_path)
                    txt_files_found += 1
        
        # Clean up repo directory
        shutil.rmtree(repo_dir)
        
        st.success(f"✅ Successfully synced {txt_files_found} documents from GitHub!")
        return True
    
    except Exception as e:
        st.error(f"❌ GitHub sync failed: {str(e)}")
        return False

def check_github_status():
    """Check GitHub token availability and repo access"""
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        return {
            "status": "missing",
            "message": "No GitHub token found",
            "color": "red"
        }
    
    try:
        # Test token by making a simple API call
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        response = requests.get(
            "https://api.github.com/repos/Umer-K/family-profiles",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                "status": "connected",
                "message": "GitHub access verified",
                "color": "green"
            }
        elif response.status_code == 404:
            return {
                "status": "not_found",
                "message": "Repository not found or no access",
                "color": "orange"
            }
        elif response.status_code == 401:
            return {
                "status": "unauthorized",
                "message": "Invalid GitHub token",
                "color": "red"
            }
        else:
            return {
                "status": "error",
                "message": f"GitHub API error: {response.status_code}",
                "color": "orange"
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Connection error: {str(e)}",
            "color": "orange"
        }


# ================= RAG SYSTEM CLASS =================
@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    return ProductionRAGSystem()

class ProductionRAGSystem:
    def __init__(self, collection_name="streamlit_rag_docs"):
        self.collection_name = collection_name
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer('all-mpnet-base-v2')
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            self.model = None
            return
            
        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            try:
                self.collection = self.client.get_collection(collection_name)
            except:
                self.collection = self.client.create_collection(collection_name)
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {e}")
            self.client = None
            return
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def get_collection_count(self):
        """Get number of documents in collection"""
        try:
            return self.collection.count() if self.collection else 0
        except:
            return 0

    def load_documents_from_folder(self, folder_path="documents"):
        """Load documents from folder"""
        if not os.path.exists(folder_path):
            return []
        
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if not txt_files:
            return []
        
        all_chunks = []
        for filename in txt_files:
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                    if content:
                        chunks = self.text_splitter.split_text(content)
                        for i, chunk in enumerate(chunks):
                            all_chunks.append({
                                'content': chunk,
                                'source_file': filename,
                                'chunk_index': i,
                                'char_count': len(chunk)
                            })
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
                return all_chunks

    def index_documents(self, document_folder="documents"):
        """Index documents with progress bar"""
        if not self.model or not self.client:
            return False
        
        chunks = self.load_documents_from_folder(document_folder)
        if not chunks:
            return False
        
        # Clear existing collection
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)
        except:
            pass
        
        # Create embeddings with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        chunk_texts = [chunk['content'] for chunk in chunks]
        
        try:
            status_text.text("Creating embeddings...")
            embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
            
            status_text.text("Storing in database...")
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{chunk['source_file']}_{chunk['chunk_index']}"
                
                metadata = {
                    "source_file": chunk['source_file'],
                    "chunk_index": chunk['chunk_index'],
                    "char_count": chunk['char_count']
                }
                
                self.collection.add(
                    documents=[chunk['content']],
                    ids=[chunk_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata]
                )
                
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
            status_text.empty()
            return True
            
        except Exception as e:
            st.error(f"Error during indexing: {e}")
            progress_bar.empty()
            status_text.empty()
            return False
            
    def expand_query_with_family_terms(self, query):
        """Expand query to include family relationship synonyms"""
        family_mappings = {
            'mother': ['mama', 'mom', 'ammi'],
            'mama': ['mother', 'mom', 'ammi'],
            'father': ['papa', 'dad', 'abbu'],
            'papa': ['father', 'dad', 'abbu'],
            'brother': ['bhai', 'bro'],
            'bhai': ['brother', 'bro'],
            'sister': ['behn', 'sis'],
            'behn': ['sister', 'sis']
        }
        
        expanded_terms = [query]
        query_lower = query.lower()
        
        for key, synonyms in family_mappings.items():
            if key in query_lower:
                for synonym in synonyms:
                    expanded_terms.append(query_lower.replace(key, synonym))
        
        return expanded_terms

    def search(self, query, n_results=5):
        """Search for relevant chunks with family relationship mapping"""
        if not self.model or not self.collection:
            return None
        
        try:
            # Expand query with family terms
            expanded_queries = self.expand_query_with_family_terms(query)
            all_results = []
            
            # Search with all expanded terms
            for search_query in expanded_queries:
                query_embedding = self.model.encode([search_query])[0].tolist()
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                
                if results['documents'][0]:
                    for chunk, distance, metadata in zip(
                        results['documents'][0],
                        results['distances'][0],
                        results['metadatas'][0]
                    ):
                        similarity = max(0, 1 - distance)
                        all_results.append({
                            'content': chunk,
                            'metadata': metadata,
                            'similarity': similarity,
                            'query_used': search_query
                        })
            
            if not all_results:
                return None
            
            # Remove duplicates and sort by similarity
            seen_chunks = set()
            unique_results = []
            for result in all_results:
                chunk_id = f"{result['metadata']['source_file']}_{result['content'][:50]}"
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append(result)
            
            # Sort by similarity and take top results
            unique_results.sort(key=lambda x: x['similarity'], reverse=True)
            search_results = unique_results[:n_results]
            
            # Debug: Show search results for troubleshooting
            print(f"Search for '{query}' (expanded to {len(expanded_queries)} terms) found {len(search_results)} results")
            for i, result in enumerate(search_results[:3]):
                print(f"  {i+1}. Similarity: {result['similarity']:.3f} | Source: {result['metadata']['source_file']} | Query: {result['query_used']}")
                print(f"     Content preview: {result['content'][:100]}...")
            
            return search_results
        
        except Exception as e:
            st.error(f"Search error: {e}")
            return None

    def extract_direct_answer(self, query, content):
        """Extract direct answer from content"""
        query_lower = query.lower()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        query_words = set(query_lower.split())
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            exact_matches = len(query_words.intersection(sentence_words))
            
            # Bonus scoring for key terms
            bonus_score = 0
            if '401k' in query_lower and ('401' in sentence.lower() or 'retirement' in sentence.lower()):
                bonus_score += 3
            if 'sick' in query_lower and 'sick' in sentence.lower():
                bonus_score += 3
            if 'vacation' in query_lower and 'vacation' in sentence.lower():
                bonus_score += 3
            
            total_score = exact_matches * 2 + bonus_score
            
            if total_score > 0:
                scored_sentences.append((sentence, total_score))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentence = scored_sentences[0][0]
            if not best_sentence.endswith('.'):
                best_sentence += '.'
            return best_sentence
        
        # Fallback
        for sentence in sentences:
            if len(sentence) > 30:
                return sentence + ('.' if not sentence.endswith('.') else '')
        
        return content[:200] + "..."

    def generate_answer(self, query, search_results, use_ai_enhancement=True, unlimited_tokens=False):
        """Generate both AI and extracted answers with proper token handling"""
        if not search_results:
            return {
                'ai_answer': "No information found in documents.",
                'extracted_answer': "No information found in documents.",
                'sources': [],
                'confidence': 0,
                'has_both': False
            }
        
        best_result = search_results[0]
        sources = list(set([r['metadata']['source_file'] for r in search_results[:2]]))
        avg_confidence = sum(r['similarity'] for r in search_results[:2]) / len(search_results[:2])
        
        # Always generate extracted answer
        extracted_answer = self.extract_direct_answer(query, best_result['content'])
        
        # Try AI answer if requested and API key available
        ai_answer = None
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        
        if use_ai_enhancement and openrouter_key:
            # Build context from search results
            context = "\n\n".join([f"Source: {r['metadata']['source_file']}\nContent: {r['content']}"
                                  for r in search_results[:3]])
            
            # Create focused prompt for rich, engaging family responses
            if unlimited_tokens:
                prompt = f"""You are a warm, caring family assistant who knows everyone well. Based on the family information below, provide a rich, detailed, and engaging response.
Family Document Context:
{context}
Question: {query}
Instructions:
- Use the document information as your foundation
- Expand with logical personality traits and qualities someone like this would have
- Add 3-4 additional lines of thoughtful insights about their character
- Use 5-6 relevant emojis throughout the response to make it warm and engaging
- Write in a caring, family-friend tone
- If someone asks about relationships (like "mother" = "mama"), make those connections
- Make the response feel personal and detailed, not just a basic fact
- Include both strengths and endearing qualities
- Keep it warm but informative (4-6 sentences total)
- Sprinkle emojis naturally throughout, not just at the end

Remember: You're helping someone learn about their family members in a meaningful way! 💝"""
                max_tokens = 400  # Increased for richer responses
                temperature = 0.3  # Slightly more creative
            else:
                # Shorter but still enhanced prompt for conservative mode
                prompt = f"""Based on this family info: {extracted_answer}
Question: {query}
Give a warm, detailed answer with 3-4 emojis spread throughout. Add 2-3 more qualities this person likely has. Make it caring and personal! 💝"""
                max_tokens = 150  # Better than 50 for family context
                temperature = 0.2
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://huggingface.co/spaces",
                        "X-Title": "RAG Chatbot"
                    },
                    json={
                        "model": "openai/gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    ai_response = response.json()['choices'][0]['message']['content'].strip()
                    ai_answer = ai_response if len(ai_response) > 10 else extracted_answer
                else:
                    # Log the actual error for debugging
                    error_detail = ""
                    try:
                        error_detail = response.json().get('error', {}).get('message', '')
                    except:
                        pass
                    
                    if response.status_code == 402:
                        st.warning("💳 OpenRouter credits exhausted. Using extracted answers only.")
                    elif response.status_code == 429:
                        st.warning("⏱️ Rate limit reached. Using extracted answers only.")
                    elif response.status_code == 401:
                        st.error("🔑 Invalid API key. Check your OpenRouter key.")
                    elif response.status_code == 400:
                        st.error(f"❌ Bad request: {error_detail}")
                    else:
                        st.warning(f"API Error {response.status_code}: {error_detail}. Using extracted answers only.")
            
            except requests.exceptions.Timeout:
                st.warning("⏱️ API timeout. Using extracted answers only.")
            except Exception as e:
                st.warning(f"API Exception: {str(e)}. Using extracted answers only.")
        
        return {
            'ai_answer': ai_answer,
            'extracted_answer': extracted_answer,
            'sources': sources,
            'confidence': avg_confidence,
            'has_both': ai_answer is not None
        }

def get_general_ai_response(query, unlimited_tokens=False):
    """Get AI response for general questions with family-friendly enhancement"""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not openrouter_key:
        return "I can only answer questions about your family members from the uploaded documents. Please add an OpenRouter API key for general conversations. 💝"
    
    try:
        # Adjust parameters based on token availability
        if unlimited_tokens:
            max_tokens = 350  # Good limit for detailed family responses
            temperature = 0.5
            prompt = f"""You are a caring family assistant. Someone is asking about their family but I couldn't find specific information in their family documents.
Question: {query}
Please provide a warm, helpful response that:
- Acknowledges I don't have specific information about their family member
- Suggests they might want to add more details to their family profiles
- Offers to help in other ways
- Uses a caring, family-friendly tone with appropriate emojis
- Keep it supportive and understanding 💝"""
        else:
            max_tokens = 100  # Reasonable for conservative mode
            temperature = 0.4
            prompt = f"Family question: {query[:100]} - I don't have info about this family member. Give a caring, helpful response with emojis 💝"
            
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://huggingface.co/spaces",
                "X-Title": "RAG Chatbot"
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            # Get detailed error information
            error_detail = ""
            try:
                error_detail = response.json().get('error', {}).get('message', '')
            except:
                pass
            
            if response.status_code == 402:
                return "Sorry, OpenRouter credits exhausted. Please add more credits or top up your account."
            elif response.status_code == 429:
                return "Rate limit reached. Please try again in a moment."
            elif response.status_code == 401:
                return "Invalid API key. Please check your OpenRouter API key configuration."
            elif response.status_code == 400:
                return f"Bad request: {error_detail}. Please try rephrasing your question."
            else:
                return f"API error (Status: {response.status_code}): {error_detail}. Please try again."
    
    except requests.exceptions.Timeout:
        return "Request timeout. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

def get_user_id():
    """Get unique ID for this user session"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]
    return st.session_state.user_id

# ================= SESSION AND HISTORY MANAGEMENT =================
def load_sessions():
    """Load all chat sessions from file"""
    if not os.path.exists(SESSIONS_FILE):
        return {}
    with open(SESSIONS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_sessions(sessions):
    """Save all chat sessions to file"""
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=4)

def load_history(session_id):
    """Load chat history for a specific session"""
    sessions = load_sessions()
    return sessions.get(session_id, [])

def save_history(session_id, history):
    """Save chat history for a specific session"""
    sessions = load_sessions()
    sessions[session_id] = history
    save_sessions(sessions)

def new_session():
    """Create a new chat session"""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_title = f"New Chat - {datetime.now().strftime('%b %d, %I:%M %p')}"
    save_history(st.session_state.session_id, [])


# ================= MAIN APP LOGIC =================
def main():
    """Main function to run the Streamlit app"""
    # Initialize RAG system and get counts
    rag_system = initialize_rag_system()
    num_indexed_chunks = rag_system.get_collection_count()

    # Initialize state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        new_session()
    
    # Sidebar UI
    with st.sidebar:
        st.title("RAG Chat Flow ✘")
        
        # Dark mode toggle
        st.markdown(
            f'<button class="dark-mode-toggle" onclick="window.parent.document.querySelector(\'#my_dark_mode_toggle_button\').click()">{ "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode" }</button>',
            unsafe_allow_html=True
        )
        if st.button("New Chat", use_container_width=True, type="primary"):
            new_session()
            st.rerun()

        st.divider()
        
        # Sync Documents section
        with st.expander("📂 Sync Documents", expanded=False):
            st.markdown(
                f'<div class="github-status"><b>GitHub Status:</b> {check_github_status()["message"]}</div>',
                unsafe_allow_html=True
            )
            if st.button("🔄 Sync from GitHub"):
                if clone_github_repo():
                    rag_system.index_documents()
                    st.success("✅ Documents indexed successfully!")
                    st.rerun()
            
            st.info("💡 Make sure you set the `GITHUB_TOKEN` secret in Streamlit Cloud.")
            st.markdown(
                f'<div class="document-status"><b>Indexed Chunks:</b> {num_indexed_chunks}</div>',
                unsafe_allow_html=True
            )
            
        st.divider()

        # Personality Questions section
        with st.expander("💬 Quick Questions", expanded=False):
            st.markdown(
                '<div class="personality-section"><b>Find out about your family members!</b></div>',
                unsafe_allow_html=True
            )
            for q in PERSONALITY_QUESTIONS:
                if st.button(q, key=q):
                    st.session_state.messages.append({"role": "user", "content": q})
                    handle_query(q)

        st.divider()
        
        # Chat History section
        st.subheader("Chat History")
        sessions = load_sessions()
        session_ids = list(sessions.keys())
        for session_id in session_ids:
            history_preview = sessions[session_id]
            if history_preview:
                first_message = history_preview[0]
                preview_text = first_message["content"][:30] + "..." if first_message else "New Chat"
                
                if st.button(f"📄 {preview_text}", key=f"session_{session_id}", use_container_width=True):
                    st.session_state.session_id = session_id
                    st.session_state.messages = sessions[session_id]
                    st.rerun()
                    
    # Main chat UI
    st.header(st.session_state.get('chat_title', "RAG Chat Flow ✘"))

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if "sources" in message:
                st.markdown(
                    f'<div class="rag-attribution"><b>Sources:</b> {", ".join(message["sources"])}</div>',
                    unsafe_allow_html=True
                )
    
    # Accept user input
    if user_query := st.chat_input("Ask about your family..."):
        handle_query(user_query)


def handle_query(query):
    """Process the user's query and generate a response"""
    rag_system = initialize_rag_system()
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        search_results = rag_system.search(query)
        
        if not search_results:
            response_content = get_general_ai_response(query)
            with st.chat_message("assistant"):
                st.markdown(response_content, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
        else:
            response_data = rag_system.generate_answer(query, search_results, use_ai_enhancement=True)
            
            with st.chat_message("assistant"):
                final_answer = response_data['ai_answer'] if response_data['has_both'] else response_data['extracted_answer']
                st.markdown(final_answer, unsafe_allow_html=True)
                
                # Display sources
                if response_data['sources']:
                    st.markdown(
                        f'<div class="rag-attribution"><b>Sources:</b> {", ".join(response_data["sources"])}</div>',
                        unsafe_allow_html=True
                    )
            
            # Save the response to session state with sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "sources": response_data["sources"]
            })
    
    # Save the updated history
    save_history(st.session_state.session_id, st.session_state.messages)


if __name__ == "__main__":
    main()
