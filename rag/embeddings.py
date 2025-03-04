"""Vector embedding utilities."""

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

from rag.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, VECTOR_STORE_PATH


def get_embeddings():
    """
    Get the embedding model.
    
    Returns:
        An embedding model instance.
    """
    try:
        # Try to use Ollama embeddings first
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBED_MODEL,
        )
        # Test the embeddings
        embeddings.embed_query("Test query")
        print(f"Using Ollama embeddings with model: {OLLAMA_EMBED_MODEL}")
        return embeddings
    except Exception as e:
        print(f"Failed to use Ollama embeddings: {e}")
        print("Falling back to local HuggingFace embeddings")
        
        # Fall back to HuggingFace embeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )


def create_vector_store(documents: List[Document], store_path: str = VECTOR_STORE_PATH):
    """
    Create a vector store from documents.
    
    Args:
        documents: List of documents to embed.
        store_path: Path to save the vector store.
        
    Returns:
        FAISS vector store instance.
    """
    embeddings = get_embeddings()
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save vector store
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    vector_store.save_local(store_path)
    
    print(f"Created vector store with {len(documents)} documents at {store_path}")
    return vector_store


def load_vector_store(store_path: str = VECTOR_STORE_PATH):
    """
    Load a vector store from disk.
    
    Args:
        store_path: Path to the vector store.
        
    Returns:
        FAISS vector store instance.
    """
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Vector store not found at {store_path}")
    
    embeddings = get_embeddings()
    # Add allow_dangerous_deserialization=True to handle the pickle security warning
    vector_store = FAISS.load_local(
        store_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print(f"Loaded vector store from {store_path}")
    return vector_store 