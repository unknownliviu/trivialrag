#!/usr/bin/env python3
"""
Script to process and index documents for the RAG pipeline.
"""

import os
import argparse
from tqdm import tqdm

from rag.document_loader import load_documents, split_documents
from rag.embeddings import create_vector_store
from rag.config import VECTOR_STORE_PATH


def main():
    """Main function to index documents."""
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing documents to index",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default=VECTOR_STORE_PATH,
        help="Path to save the vector store",
    )
    args = parser.parse_args()
    
    print(f"Indexing documents from {args.data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Creating data directory: {args.data_dir}")
        os.makedirs(args.data_dir)
        print(f"Please add documents to {args.data_dir} and run this script again.")
        return
    
    # Check if there are any documents in the data directory
    if not any(f.endswith(('.pdf', '.txt')) for f in os.listdir(args.data_dir)):
        print(f"No PDF or TXT documents found in {args.data_dir}")
        print(f"Please add documents to {args.data_dir} and run this script again.")
        return
    
    # Load documents
    print("Loading documents...")
    documents = load_documents(args.data_dir)
    
    if not documents:
        print("No documents were loaded. Please check your data directory.")
        return
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    # Create vector store
    print("Creating vector store...")
    create_vector_store(chunks, args.vector_store)
    
    print(f"Indexing complete! Vector store saved to {args.vector_store}")
    print("You can now run query.py to ask questions about your documents.")


if __name__ == "__main__":
    main() 