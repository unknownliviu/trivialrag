"""Text generation utilities using Ollama."""

from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from rag.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL


def get_llm():
    """
    Get the LLM model.
    
    Returns:
        An Ollama LLM instance.
    """
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_LLM_MODEL,
        temperature=0.1,
    )
    return llm


def create_rag_prompt():
    """
    Create a prompt template for RAG.
    
    Returns:
        A PromptTemplate instance.
    """
    template = """
You are a helpful AI assistant that answers questions based on the provided context.
If the answer cannot be found in the context, just say that you don't know based on the provided information.
Do not make up answers that are not supported by the context.

Context:
{context}

Question: {question}

Answer:
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )
    return prompt


def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer based on context and question.
    
    Args:
        context: Context string from retrieved documents.
        question: User's question.
        
    Returns:
        Generated answer.
    """
    llm = get_llm()
    prompt = create_rag_prompt()
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.run(context=context, question=question)
    return response 