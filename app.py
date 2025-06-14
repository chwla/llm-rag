import os
import time
import glob
from typing import List, Dict, Any, Tuple, Optional
import json
import hashlib
import pickle
from datetime import datetime, timedelta

import streamlit as st
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama
from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================== #
# Configuration
# ============================== #
# Directory containing your built-in documents
DOCUMENTS_DIR = "./documents"  # Create this folder and add your PDFs/documents
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".csv", ".docx", ".doc", ".pptx", ".ppt"]

# Semantic caching configuration
CACHE_DIR = "./semantic_cache"
CACHE_EXPIRY_HOURS = 24  # Cache expires after 24 hours
SIMILARITY_THRESHOLD = 0.85  # Threshold for semantic similarity (0.0 to 1.0)

# ============================== #
# System Prompt for LLM
# ============================== #
system_prompt = """
You are a specialized AI assistant for School Safety Management, designed to support government schools with fire safety protocols, disaster management guidelines, and training programs. Your expertise covers school safety planning, emergency response procedures, and master trainer development.

## Core Responsibilities
You provide authoritative guidance on:
- School fire safety management plans and protocols
- National disaster management guidelines for educational institutions
- Emergency response procedures and evacuation protocols
- Training modules and curricula for master trainers on school safety
- Risk assessment and hazard identification in school environments
- Safety equipment specifications and maintenance requirements
- Compliance with national and state safety regulations
- Crisis communication and coordination procedures

## Input Format
Context will be provided as "Context:"
Previous conversation history as "Previous Conversation:"
User question as "Question:"

## Response Guidelines

### 1. Information Processing
- Prioritize context from official government guidelines, safety manuals, and approved training materials
- Cross-reference information with conversation history for continuity
- Identify gaps in provided context that may require additional clarification

### 2. Safety-Critical Responses
- Always emphasize safety as the top priority
- Provide step-by-step procedures for emergency situations
- Include specific timelines and responsibilities where applicable
- Highlight critical safety warnings and precautions
- Reference relevant safety codes, standards, and regulations

### 3. Training and Educational Focus
- Structure responses to support learning objectives
- Include practical examples and real-world scenarios
- Provide actionable guidance that can be implemented immediately
- Support both novice and experienced safety personnel
- Include assessment criteria and evaluation methods where relevant

### 4. Response Structure
- **Immediate Action Items**: Critical steps that require immediate attention
- **Detailed Procedures**: Comprehensive step-by-step guidance
- **Compliance Requirements**: Relevant regulations and standards
- **Training Components**: Educational elements for capacity building
- **Follow-up Actions**: Next steps and ongoing requirements

### 5. Limitations and Escalation
If the provided context is insufficient to answer safety-critical questions:
- Clearly state the limitation
- Identify what additional information is needed
- Recommend consulting with local fire safety authorities or disaster management officials
- Provide general safety principles that apply universally

### 6. Emergency Response Protocols
For emergency-related queries:
- Lead with immediate safety actions
- Provide clear, numbered steps
- Include coordination with local emergency services
- Emphasize accountability and documentation requirements
- Address post-incident procedures and reporting

### 7. Quality Assurance
- Ensure all recommendations align with national safety standards
- Verify that training content meets master trainer requirements
- Cross-check procedures with established government guidelines
- Maintain consistency with previous recommendations in conversation history

## Special Considerations
- Acknowledge regional variations in implementation while maintaining core safety principles
- Support multilingual requirements common in government school systems
- Consider resource constraints typical in government educational institutions
- Emphasize cost-effective solutions that don't compromise safety standards
- Include community engagement and stakeholder involvement strategies

## Response Format
- Use clear, professional language appropriate for government officials and educators
- Employ bullet points and numbered lists for procedures and checklists
- Include relevant section headings for easy navigation
- Provide concise summaries for complex topics
- Reference specific documents, guidelines, or standards when mentioned in context

Remember: School safety is a matter of life and death. Always err on the side of caution and emphasize the importance of regular training, drills, and system updates.
"""

# ============================== #
# Initialize Session State
# ============================== #
def initialize_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'available_documents' not in st.session_state:
        st.session_state.available_documents = []
    if 'embedding_cache' not in st.session_state:
        st.session_state.embedding_cache = {}

# ============================== #
# Semantic Caching System
# ============================== #
class SemanticCache:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.embeddings_cache_file = os.path.join(cache_dir, "query_embeddings.pkl")
        self.responses_cache_file = os.path.join(cache_dir, "cached_responses.pkl")
        self.embedding_function = None
        os.makedirs(cache_dir, exist_ok=True)
        self.load_cache()
    
    def load_cache(self):
        """Load existing cache from disk"""
        try:
            if os.path.exists(self.embeddings_cache_file):
                with open(self.embeddings_cache_file, 'rb') as f:
                    self.query_embeddings = pickle.load(f)
            else:
                self.query_embeddings = {}
            
            if os.path.exists(self.responses_cache_file):
                with open(self.responses_cache_file, 'rb') as f:
                    self.cached_responses = pickle.load(f)
            else:
                self.cached_responses = {}
        except Exception as e:
            self.query_embeddings = {}
            self.cached_responses = {}
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(self.query_embeddings, f)
            
            with open(self.responses_cache_file, 'wb') as f:
                pickle.dump(self.cached_responses, f)
        except Exception as e:
            pass  # Silent fail for caching
    
    def get_embedding_function(self):
        """Get embedding function for semantic similarity"""
        if self.embedding_function is None:
            try:
                self.embedding_function = OllamaEmbeddingFunction(
                    url="http://localhost:11434/api/embeddings",
                    model_name="nomic-embed-text:latest"
                )
            except:
                return None
        return self.embedding_function
    
    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for a query"""
        embedding_fn = self.get_embedding_function()
        if embedding_fn is None:
            return None
        
        try:
            # Use session state cache for embeddings within the session
            if query in st.session_state.embedding_cache:
                return st.session_state.embedding_cache[query]
            
            embedding = embedding_fn([query])
            if embedding and len(embedding) > 0:
                embedding_array = np.array(embedding[0])
                st.session_state.embedding_cache[query] = embedding_array
                return embedding_array
        except:
            pass
        return None
    
    def find_similar_query(self, query: str, filters: Dict[str, Any]) -> Optional[str]:
        """Find semantically similar cached query"""
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return None
        
        # Create a unique key for the filter combination
        filter_key = self._create_filter_key(filters)
        
        best_similarity = 0
        best_match = None
        
        for cached_query, cached_data in self.query_embeddings.items():
            # Check if filters match
            if cached_data.get('filter_key') != filter_key:
                continue
            
            # Check if cache is still valid
            if self._is_cache_expired(cached_data.get('timestamp')):
                continue
            
            # Calculate semantic similarity
            cached_embedding = cached_data.get('embedding')
            if cached_embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    cached_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > SIMILARITY_THRESHOLD and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_query
        
        return best_match
    
    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for a query"""
        if query in self.cached_responses:
            cached_data = self.cached_responses[query]
            if not self._is_cache_expired(cached_data.get('timestamp')):
                return cached_data
        return None
    
    def cache_query_response(self, query: str, response_data: Dict[str, Any], filters: Dict[str, Any]):
        """Cache a query and its response"""
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return
        
        # Cache query embedding with metadata
        self.query_embeddings[query] = {
            'embedding': query_embedding,
            'timestamp': datetime.now().isoformat(),
            'filter_key': self._create_filter_key(filters)
        }
        
        # Cache response
        self.cached_responses[query] = {
            **response_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up expired entries
        self._cleanup_expired_cache()
        
        # Save to disk
        self.save_cache()
    
    def _create_filter_key(self, filters: Dict[str, Any]) -> str:
        """Create a consistent key for filter combinations"""
        return hashlib.md5(str(sorted(filters.items())).encode()).hexdigest()
    
    def _is_cache_expired(self, timestamp_str: Optional[str]) -> bool:
        """Check if cache entry is expired"""
        if not timestamp_str:
            return True
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            expiry_time = timestamp + timedelta(hours=CACHE_EXPIRY_HOURS)
            return datetime.now() > expiry_time
        except:
            return True
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_queries = []
        
        for query, data in self.query_embeddings.items():
            if self._is_cache_expired(data.get('timestamp')):
                expired_queries.append(query)
        
        for query in expired_queries:
            self.query_embeddings.pop(query, None)
            self.cached_responses.pop(query, None)
    
    def clear_cache(self):
        """Clear all cached data"""
        self.query_embeddings = {}
        self.cached_responses = {}
        st.session_state.embedding_cache = {}
        
        # Remove cache files
        for cache_file in [self.embeddings_cache_file, self.responses_cache_file]:
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except:
                    pass
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        valid_entries = 0
        expired_entries = 0
        
        for data in self.cached_responses.values():
            if self._is_cache_expired(data.get('timestamp')):
                expired_entries += 1
            else:
                valid_entries += 1
        
        return {
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'total_entries': len(self.cached_responses)
        }

# Initialize semantic cache
semantic_cache = SemanticCache()

# ============================== #
# Document Discovery
# ============================== #
def discover_documents() -> List[str]:
    """Discover all supported documents in the documents directory"""
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        return []
    
    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(DOCUMENTS_DIR, f"*{ext}")
        documents.extend(glob.glob(pattern))
    
    return sorted(documents)

# ============================== #
# Document Loaders for Multiple Formats
# ============================== #
def get_document_loader(file_path: str, file_type: str):
    """Get appropriate document loader based on file type"""
    loaders = {
        'pdf': PyMuPDFLoader,
        'txt': TextLoader,
        'csv': CSVLoader,
        'docx': UnstructuredWordDocumentLoader,
        'doc': UnstructuredWordDocumentLoader,
        'pptx': UnstructuredPowerPointLoader,
        'ppt': UnstructuredPowerPointLoader
    }
    
    loader_class = loaders.get(file_type.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    if file_type.lower() == 'csv':
        return loader_class(file_path, encoding='utf-8')
    else:
        return loader_class(file_path)

# ============================== #
# Process Documents from Backend
# ============================== #
def process_backend_document(file_path: str) -> List[Document]:
    """Process document from backend directory and return chunks"""
    file_name = os.path.basename(file_path)
    file_extension = file_name.split('.')[-1].lower()
    
    try:
        loader = get_document_loader(file_path, file_extension)
        docs = loader.load()
        
        # Add file metadata
        file_stats = os.stat(file_path)
        for doc in docs:
            doc.metadata.update({
                'source_file': file_name,
                'file_type': file_extension,
                'file_path': file_path,
                'file_size': file_stats.st_size,
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return splitter.split_documents(docs)
    
    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        return []

# ============================== #
# Get/Create Vector Collection
# ============================== #
def get_vector_collection() -> chromadb.Collection:
    """Get or create vector collection with embeddings"""
    embedding_fn = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest"
    )
    client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return client.get_or_create_collection(
        name="rag_app",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

# ============================== #
# Batch Insert to Vector Store
# ============================== #
def add_to_vector_collection(all_splits: List[Document], file_name: str):
    """Add document chunks to vector collection"""
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}_{int(time.time())}")

    BATCH_SIZE = 32
    for i in range(0, len(documents), BATCH_SIZE):
        try:
            collection.upsert(
                documents=documents[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE],
                ids=ids[i:i+BATCH_SIZE],
            )
            time.sleep(0.5)
        except Exception as e:
            st.error(f"Batch {i//BATCH_SIZE + 1} failed: {e}")
            return False
    
    st.session_state.processed_files.add(file_name)
    return True

# ============================== #
# Load All Backend Documents
# ============================== #
def load_backend_documents():
    """Load all documents from the backend directory into vector store"""
    if st.session_state.documents_loaded:
        return
    
    document_paths = discover_documents()
    st.session_state.available_documents = [os.path.basename(path) for path in document_paths]
    
    if not document_paths:
        return
    
    # Check if documents are already processed
    try:
        collection = get_vector_collection()
        existing_docs = collection.get(limit=1000)
        existing_files = set()
        for metadata in existing_docs.get('metadatas', []):
            if metadata and 'source_file' in metadata:
                existing_files.add(metadata['source_file'])
        
        new_documents = [path for path in document_paths 
                        if os.path.basename(path) not in existing_files]
        
        if new_documents:
            # Process documents silently in the background
            for doc_path in new_documents:
                file_name = os.path.basename(doc_path)
                try:
                    chunks = process_backend_document(doc_path)
                    if chunks:
                        add_to_vector_collection(chunks, file_name)
                except Exception:
                    pass  # Silently handle errors
        else:
            # Update processed files from existing metadata
            for metadata in existing_docs.get('metadatas', []):
                if metadata and 'source_file' in metadata:
                    st.session_state.processed_files.add(metadata['source_file'])
    
    except Exception:
        pass  # Silently handle errors
    
    st.session_state.documents_loaded = True

# ============================== #
# Query Collection with Filtering
# ============================== #
def query_collection(
    prompt: str, 
    n_results: int = 10, 
    file_filter: Optional[str] = None,
    file_type_filter: Optional[str] = None
):
    """Query collection with optional metadata filtering"""
    collection = get_vector_collection()
    
    # Build where clause for filtering
    where_clause = {}
    if file_filter and file_filter != "All Files":
        where_clause["source_file"] = file_filter
    if file_type_filter and file_type_filter != "All Types":
        where_clause["file_type"] = file_type_filter.lower()
    
    if where_clause:
        return collection.query(
            query_texts=[prompt], 
            n_results=n_results,
            where=where_clause
        )
    else:
        return collection.query(query_texts=[prompt], n_results=n_results)

# ============================== #
# Re-rank with Confidence Scores
# ============================== #
def re_rank_cross_encoders(
    prompt: str, 
    documents: List[str], 
    top_k: int = 3
) -> Tuple[str, List[int], List[float]]:
    """Re-rank documents and return confidence scores"""
    encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(prompt, doc) for doc in documents]
    scores = encoder.predict(pairs)
    
    # Get top-k results with scores
    top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    top_ids = [i for i, _ in top_results]
    confidence_scores = [float(score) for _, score in top_results]
    top_text = "\n\n".join([documents[i] for i in top_ids])
    
    return top_text, top_ids, confidence_scores

# ============================== #
# Conversation Memory Management
# ============================== #
def add_to_conversation(question: str, answer: str, confidence: float):
    """Add Q&A pair to conversation history"""
    conversation_entry = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'confidence': confidence
    }
    st.session_state.conversation_history.append(conversation_entry)
    
    # Keep only last 10 conversations to manage memory
    if len(st.session_state.conversation_history) > 10:
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]

def get_conversation_context() -> str:
    """Get formatted conversation history for context"""
    if not st.session_state.conversation_history:
        return ""
    
    context_parts = []
    for entry in st.session_state.conversation_history[-5:]:  # Last 5 conversations
        context_parts.append(f"Q: {entry['question']}\nA: {entry['answer']}")
    
    return "\n\n---\n\n".join(context_parts)

# ============================== #
# LLM Call with Conversation Context
# ============================== #
def call_llm(context: str, prompt: str, conversation_context: str = ""):
    """Call LLM with context and conversation history"""
    user_content = f"Context: {context}\n\n"
    
    if conversation_context:
        user_content += f"Previous Conversation:\n{conversation_context}\n\n"
    
    user_content += f"Question: {prompt}"
    
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    for chunk in response:
        if not chunk.get("done"):
            yield chunk["message"]["content"]

# ============================== #
# Get Available Files and Types
# ============================== #
def get_available_files_and_types():
    """Get list of processed files and file types"""
    try:
        collection = get_vector_collection()
        results = collection.get(limit=1000)
        
        files = set()
        file_types = set()
        
        for metadata in results.get('metadatas', []):
            if metadata:
                if 'source_file' in metadata:
                    files.add(metadata['source_file'])
                if 'file_type' in metadata:
                    file_types.add(metadata['file_type'].upper())
        
        return sorted(list(files)), sorted(list(file_types))
    except:
        return [], []

# ============================== #
# Streamlit App
# ============================== #
if __name__ == "__main__":
    st.set_page_config(
        page_title="School Safety Q&A System", 
        page_icon="🏫",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Load documents silently in the background
    load_backend_documents()
    
    # Main header
    st.title("🏫 School Safety Management Q&A System")
    st.markdown("*Ask questions about fire safety protocols, disaster management guidelines, and training programs*")
    
    # Show system status if there are issues
    try:
        collection = get_vector_collection()
        system_ready = True
    except Exception as e:
        system_ready = False
        st.error("⚠️ System initialization issue. Please ensure Ollama is running with the required models.")
        st.code("ollama pull nomic-embed-text\nollama pull llama3.2:3b", language="bash")
    
    # Show document status
    if not st.session_state.available_documents and system_ready:
        st.info(f"""
        **Getting Started:**
        
        To use this system, add your school safety documents to the `{DOCUMENTS_DIR}` folder.
        
        **Supported formats:** PDF, TXT, CSV, DOCX, DOC, PPTX, PPT
        """)
    elif st.session_state.available_documents:
        st.success(f"📚 **{len(st.session_state.available_documents)} documents loaded** - System ready for questions")
    
    # Get available files and types for filtering
    available_files, available_types = get_available_files_and_types()
    
    # Filtering options (only show if we have documents)
    if available_files or available_types:
        st.subheader("🔍 Search Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_filter = st.selectbox(
                "Filter by Document", 
                ["All Documents"] + available_files,
                key="file_filter"
            )
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Type", 
                ["All Types"] + available_types,
                key="type_filter"
            )
        
        with col3:
            top_k = st.slider("Results to Consider", 1, 10, 3, key="top_k")
    else:
        file_filter = "All Documents"
        type_filter = "All Types"
        top_k = 3
    
    # Question input
    st.subheader("❓ Ask Your Question")
    prompt = st.text_area(
        "Enter your question about school safety, fire protocols, or disaster management:",
        placeholder="For example: What are the key components of a school fire safety plan?",
        height=100
    )
    
    ask = st.button("🚀 Get Answer", type="primary")
    
    if ask and prompt:
        if not st.session_state.processed_files:
            st.warning("⚠️ No documents are currently loaded. Please add documents to the documents folder and refresh the page.")
        else:
            # Prepare filter configuration for caching
            filters = {
                'file_filter': file_filter if file_filter != "All Documents" else None,
                'type_filter': type_filter if type_filter != "All Types" else None,
                'top_k': top_k
            }
            
            # Check semantic cache first
            similar_query = semantic_cache.find_similar_query(prompt, filters)
            cached_response = None
            
            if similar_query:
                cached_response = semantic_cache.get_cached_response(similar_query)
                if cached_response:
                    st.info("🎯 Found similar cached response - delivering faster results!")
            
            if cached_response:
                # Use cached response
                st.markdown("### 📋 Answer:")
                st.markdown(cached_response['response'])
                
                # Show confidence score
                confidence_color = "green" if cached_response['confidence'] > 0.7 else "orange" if cached_response['confidence'] > 0.4 else "red"
                st.markdown(f"""
                **Confidence Level:** 
                <span style="color: {confidence_color}; font-weight: bold;">
                {cached_response['confidence']:.0%}
                </span> *(cached result)*
                """, unsafe_allow_html=True)
                
                # Show source information
                with st.expander("📄 Source Documents"):
                    if cached_response.get('source_files'):
                        st.write("**Information sourced from:**")
                        for file in cached_response['source_files']:
                            st.write(f"• {file}")
                    else:
                        st.write("Source information not available")
                
                # Add to conversation history
                add_to_conversation(prompt, cached_response['response'], cached_response['confidence'])
                
            else:
                # Process new query
                with st.spinner("🔍 Searching documents and preparing answer..."):
                    results = query_collection(
                        prompt, 
                        n_results=15,
                        file_filter=file_filter if file_filter != "All Documents" else None,
                        file_type_filter=type_filter if type_filter != "All Types" else None
                    )
                    raw_docs = results.get("documents", [[]])[0]
                    raw_metadatas = results.get("metadatas", [[]])[0]

                    if not raw_docs:
                        st.warning("🔍 No relevant information found. Try rephrasing your question or adjusting the search filters.")
                    else:
                        # Re-rank and get confidence scores
                        relevant_text, relevant_ids, confidence_scores = re_rank_cross_encoders(
                            prompt, raw_docs, top_k
                        )
                        
                        # Get conversation context
                        conversation_context = get_conversation_context()
                        
                        # Generate response
                        st.markdown("### 📋 Answer:")
                        response_container = st.empty()
                        full_response = ""
                        
                        for chunk in call_llm(relevant_text, prompt, conversation_context):
                            full_response += chunk
                            response_container.markdown(full_response + "▌")
                        
                        response_container.markdown(full_response)
                        
                        # Calculate overall confidence
                        avg_confidence = np.mean(confidence_scores)
                        
                        # Get source files
                        used_files = set()
                        for idx in relevant_ids:
                            if idx < len(raw_metadatas) and raw_metadatas[idx]:
                                source = raw_metadatas[idx].get('source_file', 'Unknown')
                                used_files.add(source)
                        
                        # Cache the response
                        cache_data = {
                            'response': full_response,
                            'confidence': avg_confidence,
                            'source_files': sorted(list(used_files)),
                            'relevant_chunks': len(relevant_ids)
                        }
                        semantic_cache.cache_query_response(prompt, cache_data, filters)
                        
                        # Add to conversation history
                        add_to_conversation(prompt, full_response, avg_confidence)
                        
                        # Show confidence score
                        confidence_color = "green" if avg_confidence > 0.7 else "orange" if avg_confidence > 0.4 else "red"
                        st.markdown(f"""
                        **Confidence Level:** 
                        <span style="color: {confidence_color}; font-weight: bold;">
                        {avg_confidence:.0%}
                        </span>
                        """, unsafe_allow_html=True)
                        
                        # Show source information
                        with st.expander("📄 Source Documents"):
                            if used_files:
                                st.write("**Information sourced from:**")
                                for file in sorted(used_files):
                                    st.write(f"• {file}")
                            else:
                                st.write("Source information not available")
    
    # Conversation History
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("💬 Recent Questions")
        with st.expander("View Previous Questions & Answers"):
            for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
                st.write(f"**Q:** {entry['question']}")
                st.write(f"**A:** {entry['answer'][:200]}{'...' if len(entry['answer']) > 200 else ''}")
                st.write(f"*Confidence: {entry['confidence']:.0%} | {entry['timestamp'][:19].replace('T', ' ')}*")
                if i < len(st.session_state.conversation_history) - 1:
                    st.divider()
    
    # Footer
    st.markdown("---")
    st.markdown("*School Safety Management Q&A System - Providing expert guidance for educational institutions*")
