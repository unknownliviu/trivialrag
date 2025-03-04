#!/usr/bin/env python3
"""
Script to query the RAG pipeline.
"""

import os
import argparse
import time

from rag.embeddings import load_vector_store
from rag.retriever import retrieve_documents, format_context
from rag.generator import generate_answer
from rag.config import VECTOR_STORE_PATH, OLLAMA_LLM_MODEL


def main():
    """Main function to query the RAG pipeline."""
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "--vector_store",
        type=str,
        default=VECTOR_STORE_PATH,
        help="Path to the vector store",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run (if not in interactive mode)",
    )
    args = parser.parse_args()
    
    # Check if vector store exists
    if not os.path.exists(args.vector_store):
        print(f"Vector store not found at {args.vector_store}")
        print("Please run index_documents.py first to create the vector store.")
        return
    
    # Load vector store
    print(f"Loading vector store from {args.vector_store}...")
    vector_store = load_vector_store(args.vector_store)
    
    print(f"Using LLM model: {OLLAMA_LLM_MODEL}")
    
    if args.interactive:
        print("\n=== Interactive RAG Query Mode ===")
        print("Type 'exit' or 'quit' to end the session.")
        
        while True:
            query = input("\nEnter your question: ")
            
            if query.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            if not query.strip():
                continue
            
            start_time = time.time()
            
            # Retrieve relevant documents
            documents = retrieve_documents(vector_store, query)
            
            # Format context
            context = format_context(documents)
            
            # Generate answer
            print("\nGenerating answer...")
            answer = generate_answer(context, query)
            
            end_time = time.time()
            
            print(f"\nAnswer: {answer}")
            print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    
    elif args.query:
        query = args.query
        
        start_time = time.time()
        
        # Retrieve relevant documents
        documents = retrieve_documents(vector_store, query)
        
        # Format context
        context = format_context(documents)
        
        # Generate answer
        print("\nGenerating answer...")
        answer = generate_answer(context, query)
        
        end_time = time.time()
        
        print(f"\nAnswer: {answer}")
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    
    else:
        print("Please provide a query with --query or use --interactive mode.")


if __name__ == "__main__":
    main() 