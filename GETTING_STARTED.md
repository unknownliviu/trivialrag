# Getting Started with Local RAG Pipeline

This guide will help you set up and use the local RAG pipeline with Ollama.

## Prerequisites

Before you begin, make sure you have:

1. Python 3.8+ installed
2. [Ollama](https://ollama.ai/download) installed and running

## Setup

1. **Check Ollama Status**

   Run the following command to check if Ollama is installed and running:

   ```bash
   python check_ollama.py
   ```

   If Ollama is not installed or running, follow the instructions provided by the script.

2. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Choose Models**

   Run the model info script to see recommended models for your RAG pipeline:

   ```bash
   python model_info.py
   ```

   Install the recommended models using Ollama:

   ```bash
   ollama pull nomic-embed-text  # For embeddings
   ollama pull llama3            # For text generation
   ```

   You can customize the models by editing the `.env` file.

## Usage

### 1. Prepare Your Documents

Place your text documents (PDF, TXT) in the `data/` directory. A sample document is already provided for testing.

### 2. Index Your Documents

Run the indexing script to process and index your documents:

```bash
python index_documents.py
```

This will:
- Load documents from the `data/` directory
- Split them into chunks
- Create embeddings
- Store them in a vector database

### 3. Query Your Documents

You can query your documents in two ways:

**Interactive Mode**:

```bash
python query.py --interactive
```

This will start an interactive session where you can ask multiple questions.

**Single Query Mode**:

```bash
python query.py --query "What are the benefits of RAG?"
```

This will process a single query and exit.

## Customization

You can customize the RAG pipeline by editing the `.env` file:

- Change the embedding or LLM models
- Adjust chunk size and overlap
- Modify the number of retrieved documents

## Troubleshooting

If you encounter issues:

1. Make sure Ollama is running (`ollama serve`)
2. Check that you have the required models installed (`ollama list`)
3. Verify that your documents are in the correct format and location
4. Check the Python dependencies are installed correctly

## Next Steps

- Add more documents to improve the knowledge base
- Experiment with different models for embeddings and generation
- Adjust the chunking parameters for your specific documents
- Integrate the RAG pipeline into your applications 