"""Document loading and parsing utilities."""

import os
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(directory_path: str) -> List[Document]:
    """
    Load documents from a directory.
    
    Args:
        directory_path: Path to the directory containing documents.
        
    Returns:
        List of loaded documents.
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Create loaders for different file types
    pdf_loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    
    text_loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    
    # Load documents
    pdf_docs = pdf_loader.load() if os.path.exists(directory_path) else []
    text_docs = text_loader.load() if os.path.exists(directory_path) else []
    
    # Combine all documents
    all_docs = pdf_docs + text_docs
    
    print(f"Loaded {len(all_docs)} documents")
    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks.
    
    Args:
        documents: List of documents to split.
        
    Returns:
        List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks 