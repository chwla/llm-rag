import os
import time
import glob
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime

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

# ============================== #
# Configuration
# ============================== #
# Directory containing your built-in documents
DOCUMENTS_DIR = "./documents"  # Create this folder and add your PDFs/documents
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".csv", ".docx", ".doc", ".pptx", ".ppt"]

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
        st.warning("⚠️ No documents found in the documents directory. Please add some documents to get started.")
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
            progress_bar = st.progress(0)
            st.info(f"📚 Loading {len(new_documents)} new documents...")
            
            for i, doc_path in enumerate(new_documents):
                file_name = os.path.basename(doc_path)
                try:
                    chunks = process_backend_document(doc_path)
                    if chunks:
                        success = add_to_vector_collection(chunks, file_name)
                        if success:
                            st.success(f"✅ Loaded: {file_name}")
                    else:
                        st.warning(f"⚠️ No content extracted from: {file_name}")
                except Exception as e:
                    st.error(f"❌ Failed to load {file_name}: {e}")
                
                progress_bar.progress((i + 1) / len(new_documents))
            
            progress_bar.empty()
        else:
            st.info("📚 All documents are already loaded in the vector store.")
            # Update processed files from existing metadata
            for metadata in existing_docs.get('metadatas', []):
                if metadata and 'source_file' in metadata:
                    st.session_state.processed_files.add(metadata['source_file'])
    
    except Exception as e:
        st.error(f"Error checking existing documents: {e}")
    
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
# Clear Vector Store
# ============================== #
def clear_vector_store():
    """Clear all data from vector store"""
    try:
        client = chromadb.PersistentClient(path="./demo-rag-chroma")
        client.delete_collection("rag_app")
        st.session_state.processed_files.clear()
        st.session_state.conversation_history.clear()
        st.session_state.documents_loaded = False
        st.success("🗑️ Vector store cleared successfully!")
        return True
    except Exception as e:
        st.error(f"Error clearing vector store: {e}")
        return False

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
        page_title="Document Q&A System", 
        page_icon="📚",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Sidebar for system status and document management
    st.sidebar.header("📊 System Status")
    
    # Ollama status check
    try:
        collection = get_vector_collection()
        st.sidebar.success("🤖 Models Ready")
    except Exception as e:
        st.sidebar.error("❌ Model Issue")
        st.sidebar.write(f"Error: {str(e)[:50]}...")
        st.sidebar.markdown("**Quick fix:**")
        st.sidebar.code("ollama pull nomic-embed-text", language="bash")
    
    # Document loading section
    st.sidebar.subheader("📁 Document Library")
    
    # Load backend documents automatically
    with st.sidebar:
        if st.button("🔄 Process Documents", type="secondary"):
            st.session_state.documents_loaded = False
            st.session_state.processed_files.clear()
        
        load_backend_documents()
    
    # Show document statistics
    if st.session_state.available_documents:
        st.sidebar.success(f"📚 {len(st.session_state.available_documents)} documents available")
        with st.sidebar.expander("📋 Document List"):
            for doc in st.session_state.available_documents:
                status = "✅" if doc in st.session_state.processed_files else "⏳"
                st.write(f"{status} {doc}")
    else:
        st.sidebar.info("📂 No documents found")
        st.sidebar.markdown(f"**Add documents to:** `{DOCUMENTS_DIR}/`")
    
    # Clear vector store option
    if st.sidebar.button("🗑️ Clear Vector Store", type="secondary"):
        clear_vector_store()
    
    # Main app
    st.header("📚 Document Q&A System")
    st.markdown("*Ask questions from your document library with advanced filtering and conversation memory*")
    
    # Show instructions if no documents
    if not st.session_state.available_documents:
        st.info(f"""
        **Getting Started:**
        1. Create a folder named `{DOCUMENTS_DIR}` in your project directory
        2. Add your PDF, TXT, CSV, DOCX, or PPTX files to this folder
        3. Click the "🔄 Refresh Documents" button in the sidebar
        4. Start asking questions!
        
        **Supported formats:** PDF, TXT, CSV, DOCX, DOC, PPTX, PPT
        """)
    
    # Get available files and types for filtering
    available_files, available_types = get_available_files_and_types()
    
    # Filtering options (only show if we have documents)
    if available_files or available_types:
        st.subheader("🔍 Search Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_filter = st.selectbox(
                "Filter by File", 
                ["All Files"] + available_files,
                key="file_filter"
            )
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Type", 
                ["All Types"] + available_types,
                key="type_filter"
            )
        
        with col3:
            top_k = st.slider("Top Results", 1, 10, 3, key="top_k")
    else:
        file_filter = "All Files"
        type_filter = "All Types"
        top_k = 3
    
    # Question input
    st.subheader("❓ Ask Your Question")
    prompt = st.text_area(
        "**Enter your question here:**",
        placeholder="What would you like to know from your documents?",
        height=100
    )
    
    ask = st.button("🚀 Ask", type="primary")
    
    if ask and prompt:
        if not st.session_state.processed_files:
            st.warning("⚠️ No documents are loaded. Please add documents to the documents folder and refresh.")
        else:
            with st.spinner("🔍 Searching and analyzing documents..."):
                results = query_collection(
                    prompt, 
                    n_results=15,
                    file_filter=file_filter if file_filter != "All Files" else None,
                    file_type_filter=type_filter if type_filter != "All Types" else None
                )
                raw_docs = results.get("documents", [[]])[0]
                raw_metadatas = results.get("metadatas", [[]])[0]

                if not raw_docs:
                    st.warning("🔍 No relevant documents found. Try adjusting your filters or question.")
                else:
                    # Re-rank and get confidence scores
                    relevant_text, relevant_ids, confidence_scores = re_rank_cross_encoders(
                        prompt, raw_docs, top_k
                    )
                    
                    # Get conversation context
                    conversation_context = get_conversation_context()
                    
                    # Generate response
                    st.markdown("### 🤖 Answer:")
                    response_container = st.empty()
                    full_response = ""
                    
                    for chunk in call_llm(relevant_text, prompt, conversation_context):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                    
                    response_container.markdown(full_response)
                    
                    # Calculate overall confidence
                    avg_confidence = np.mean(confidence_scores)
                    
                    # Add to conversation history
                    add_to_conversation(prompt, full_response, avg_confidence)
                    
                    # Show confidence score
                    confidence_color = "green" if avg_confidence > 0.7 else "orange" if avg_confidence > 0.4 else "red"
                    st.markdown(f"""
                    **Confidence Score:** 
                    <span style="color: {confidence_color}; font-weight: bold;">
                    {avg_confidence:.2%}
                    </span>
                    """, unsafe_allow_html=True)
                    
                    # Expandable sections for details
                    with st.expander("📊 Retrieval Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Individual Confidence Scores:**")
                            for i, score in enumerate(confidence_scores):
                                st.write(f"Chunk {i+1}: {score:.2%}")
                        
                        with col2:
                            st.write("**Source Files:**")
                            used_files = set()
                            for idx in relevant_ids:
                                if idx < len(raw_metadatas) and raw_metadatas[idx]:
                                    source = raw_metadatas[idx].get('source_file', 'Unknown')
                                    used_files.add(source)
                            for file in used_files:
                                st.write(f"📄 {file}mock ")
                    
                    with st.expander("📑 Retrieved Chunks"):
                        for i, (doc, metadata) in enumerate(zip([raw_docs[idx] for idx in relevant_ids], 
                                                              [raw_metadatas[idx] for idx in relevant_ids])):
                            st.write(f"**Relevant Chunk {i+1} (Score: {confidence_scores[i]:.2%}):**")
                            if metadata:
                                st.write(f"*Source: {metadata.get('source_file', 'Unknown')}*")
                            st.write(doc)
                            st.divider()
    
    # Conversation History
    if st.session_state.conversation_history:
        st.subheader("💬 Conversation History")
        with st.expander("View Previous Questions & Answers"):
            for i, entry in enumerate(reversed(st.session_state.conversation_history)):
                st.write(f"**Q{len(st.session_state.conversation_history)-i}:** {entry['question']}")
                st.write(f"**A:** {entry['answer']}")
                st.write(f"*Confidence: {entry['confidence']:.2%} | Time: {entry['timestamp'][:19]}*")
                st.divider()
    
    # Footer
    st.markdown("---")
    st.markdown("*Document Q&A System - Powered by Local Embeddings & LLM*")
