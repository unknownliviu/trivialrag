# Local RAG Pipeline with Ollama

This project implements a fully local Retrieval-Augmented Generation (RAG) pipeline using Ollama. It can parse text documents and provide contextually relevant answers based on the document content.

## Features

- Fully local operation - no data leaves your machine
- Document parsing for various text formats (PDF, TXT)
- Vector embedding and storage using FAISS
- Retrieval-augmented generation using Ollama models

## Prerequisites

- [Ollama](https://ollama.ai/) installed locally
- Python 3.8+

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure Ollama is running on your machine

## Usage

1. Place your documents in the `data/` directory
2. Run the indexing script to process documents:
   ```
   python index_documents.py
   ```
3. Start the query interface:
   ```
   python query.py
   ```

## Project Structure

- `index_documents.py`: Script to process and index documents
- `query.py`: Script to query the indexed documents
- `rag/`: Core RAG implementation
  - `document_loader.py`: Document loading and parsing
  - `embeddings.py`: Vector embedding utilities
  - `retriever.py`: Document retrieval logic
  - `generator.py`: Text generation with Ollama
- `data/`: Directory for storing documents
- `vector_store/`: Directory for storing vector indices

## Model Selection

The default configuration uses the `llama3` model for generation and `nomic-embed-text` for embeddings, but you can configure other models in the `.env` file. 