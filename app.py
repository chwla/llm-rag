import os
import time
import tempfile
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
from streamlit.runtime.uploaded_file_manager import UploadedFile

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama
from sentence_transformers import CrossEncoder
import numpy as np

# ============================== #
# System Prompt for LLM
# ============================== #
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context and conversation history. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
Conversation history will be passed as "Previous Conversation:"
User question will be passed as "Question:"

To answer the question:
1. Consider the conversation history for context and follow-up questions.
2. Analyze the context, identifying relevant info.
3. Plan your response with logical flow.
4. Answer comprehensively using only context and conversation history.
5. If context is insufficient, say so clearly.
6. Reference previous conversation when relevant for follow-up questions.

Format:
- Use clear paragraphs.
- Use lists or headings if needed.
- Avoid outside knowledge beyond provided context.
- For follow-up questions, acknowledge the previous context.
"""

# ============================== #
# Initialize Session State
# ============================== #
def initialize_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'last_response_confidence' not in st.session_state:
        st.session_state.last_response_confidence = None

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
# Process Multiple File Formats
# ============================== #
def process_document(uploaded_file: UploadedFile) -> List[Document]:
    """Process uploaded document and return chunks"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with tempfile.NamedTemporaryFile(
        "wb", 
        suffix=f".{file_extension}", 
        delete=False
    ) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        loader = get_document_loader(temp_path, file_extension)
        docs = loader.load()
        
        # Add file metadata
        for doc in docs:
            doc.metadata.update({
                'source_file': uploaded_file.name,
                'file_type': file_extension,
                'upload_time': datetime.now().isoformat(),
                'file_size': len(uploaded_file.getvalue())
            })
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Increased for better context
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return splitter.split_documents(docs)
    
    finally:
        os.unlink(temp_path)

# ============================== #
# Get/Create Vector Collection
# ============================== #
def get_vector_collection() -> chromadb.Collection:
    """Get or create vector collection with embeddings"""
    # Use your original working model configuration
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
    st.success(f"✅ {file_name} added to vector store!")
    return True

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
        # Get a sample of metadata to determine available files and types
        results = collection.get(limit=1000)  # Get more metadata
        
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
        page_title="Enhanced RAG Q&A", 
        page_icon="📚",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Sidebar for file upload and management
    st.sidebar.header("📁 Document Management")
    
    # Simple Ollama status check
    try:
        # Test embedding function
        collection = get_vector_collection()
        st.sidebar.success("🤖 Models Ready")
    except Exception as e:
        st.sidebar.error("❌ Model Issue")
        st.sidebar.write(f"Error: {str(e)[:50]}...")
        st.sidebar.markdown("**Quick fix:**")
        st.sidebar.code("ollama pull nomic-embed-text", language="bash")
    
    # File upload
    uploaded_files = st.sidebar.file_uploader(
        "**Upload Documents**", 
        type=["pdf", "txt", "csv", "docx", "doc", "pptx", "ppt"],
        accept_multiple_files=True
    )
    
    col1, col2 = st.sidebar.columns(2)
    process = col1.button("📤 Process", type="primary")
    clear_store = col2.button("🗑️ Clear Store", type="secondary")
    
    # Process uploaded files
    if uploaded_files and process:
        progress_bar = st.sidebar.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            
            if file_name not in st.session_state.processed_files:
                try:
                    chunks = process_document(uploaded_file)
                    add_to_vector_collection(chunks, uploaded_file.name)
                except Exception as e:
                    st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")
            else:
                st.sidebar.info(f"📄 {uploaded_file.name} already processed")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
    
    # Clear vector store
    if clear_store:
        clear_vector_store()
    
    # Show processed files
    if st.session_state.processed_files:
        st.sidebar.subheader("📋 Processed Files")
        for file in st.session_state.processed_files:
            st.sidebar.text(f"✅ {file}")
    
    # Main app
    st.header("📚 Enhanced RAG Question & Answer")
    st.markdown("*Ask questions from your uploaded documents with advanced filtering and conversation memory*")
    
    # Get available files and types for filtering
    available_files, available_types = get_available_files_and_types()
    
    # Filtering options (only show if we have documents)
    if available_files or available_types or st.session_state.processed_files:
        st.subheader("🔍 Filters")
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
        # Set default values if no documents
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
        # Check if we have any processed documents
        has_documents = bool(st.session_state.processed_files)
        
        # If session state is empty, try checking vector store directly
        if not has_documents:
            try:
                collection = get_vector_collection()
                result = collection.get(limit=1)  # Try to get just one document
                has_documents = len(result.get('documents', [])) > 0
            except:
                has_documents = False
        
        if not has_documents:
            st.warning("⚠️ Please upload and process some documents first!")
        else:
            with st.spinner("🔍 Searching and analyzing documents..."):
                # Query with filters
                file_filter_val = file_filter if 'file_filter' in locals() else None
                type_filter_val = type_filter if 'type_filter' in locals() else None
                
                results = query_collection(
                    prompt, 
                    n_results=15,  # Get more results for better re-ranking
                    file_filter=file_filter_val,
                    file_type_filter=type_filter_val
                )
                raw_docs = results.get("documents", [[]])[0]
                raw_metadatas = results.get("metadatas", [[]])[0]

                if not raw_docs:
                    st.warning("🔍 No relevant documents found. Try adjusting your filters or question.")
                else:
                    # Re-rank and get confidence scores
                    top_k_val = top_k if 'top_k' in locals() else 3
                    relevant_text, relevant_ids, confidence_scores = re_rank_cross_encoders(
                        prompt, raw_docs, top_k_val
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
                    st.session_state.last_response_confidence = avg_confidence
                    
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
                                st.write(f"📄 {file}")
                    
                    with st.expander("📑 Retrieved Chunks"):
                        for i, (doc, metadata) in enumerate(zip(raw_docs, raw_metadatas)):
                            st.write(f"**Chunk {i+1}:**")
                            if metadata:
                                st.write(f"*Source: {metadata.get('source_file', 'Unknown')}*")
                            st.write(doc)
                            st.divider()
                    
                    with st.expander("⭐ Most Relevant Chunks"):
                        for i, idx in enumerate(relevant_ids):
                            st.write(f"**Top Chunk {i+1} (Score: {confidence_scores[i]:.2%}):**")
                            if idx < len(raw_metadatas) and raw_metadatas[idx]:
                                st.write(f"*Source: {raw_metadatas[idx].get('source_file', 'Unknown')}*")
                            st.write(raw_docs[idx])
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
    st.markdown("*Enhanced RAG Application with Multi-format Support, Filtering, Conversation Memory & Confidence Scoring*")
