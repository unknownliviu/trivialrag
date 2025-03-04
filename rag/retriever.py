"""Document retrieval utilities."""

from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from rag.config import TOP_K_RETRIEVAL


def get_retriever(vector_store: FAISS):
    """
    Get a retriever from a vector store.
    
    Args:
        vector_store: FAISS vector store instance.
        
    Returns:
        A retriever instance.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVAL},
    )
    return retriever


def retrieve_documents(vector_store: FAISS, query: str) -> List[Document]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        vector_store: FAISS vector store instance.
        query: Query string.
        
    Returns:
        List of retrieved documents.
    """
    retriever = get_retriever(vector_store)
    documents = retriever.get_relevant_documents(query)
    
    print(f"Retrieved {len(documents)} documents for query: {query}")
    return documents


def format_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into a context string.
    
    Args:
        documents: List of retrieved documents.
        
    Returns:
        Formatted context string.
    """
    context = "\n\n".join([doc.page_content for doc in documents])
    return context 