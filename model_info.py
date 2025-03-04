#!/usr/bin/env python3
"""
Script to list available Ollama models and provide recommendations for RAG.
"""

import argparse
import json
import requests
from tabulate import tabulate

# Add tabulate to requirements.txt
with open("requirements.txt", "a") as f:
    f.write("tabulate>=0.9.0\n")


def get_ollama_models(base_url="http://localhost:11434"):
    """
    Get list of available Ollama models.
    
    Args:
        base_url: Ollama API base URL.
        
    Returns:
        List of model information dictionaries.
    """
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running on your machine.")
        return []


def recommend_models():
    """
    Provide recommendations for RAG models.
    
    Returns:
        Dictionary of recommended models.
    """
    recommendations = {
        "embedding": {
            "best": [
                {
                    "name": "nomic-embed-text",
                    "description": "High-quality text embeddings optimized for RAG",
                    "size": "Small (268MB)",
                    "notes": "Best balance of quality and performance"
                },
                {
                    "name": "mxbai-embed-large",
                    "description": "High-quality multilingual embeddings",
                    "size": "Large (1.5GB)",
                    "notes": "Excellent for multilingual content"
                }
            ],
            "alternatives": [
                {
                    "name": "all-minilm",
                    "description": "Lightweight embedding model",
                    "size": "Very small (120MB)",
                    "notes": "Good for resource-constrained environments"
                }
            ]
        },
        "llm": {
            "best": [
                {
                    "name": "llama3",
                    "description": "Latest Llama model from Meta",
                    "size": "Medium (4.7GB)",
                    "notes": "Excellent general-purpose model"
                },
                {
                    "name": "mistral",
                    "description": "Efficient and powerful open model",
                    "size": "Medium (4.1GB)",
                    "notes": "Great balance of quality and speed"
                }
            ],
            "alternatives": [
                {
                    "name": "phi3",
                    "description": "Microsoft's efficient small model",
                    "size": "Small (1.7GB)",
                    "notes": "Good for resource-constrained environments"
                },
                {
                    "name": "llama3:8b",
                    "description": "Smaller Llama3 variant",
                    "size": "Small (4.7GB)",
                    "notes": "Faster than the larger variant"
                },
                {
                    "name": "gemma:7b",
                    "description": "Google's lightweight model",
                    "size": "Small (4.8GB)",
                    "notes": "Good performance for its size"
                }
            ]
        }
    }
    return recommendations


def main():
    """Main function to display model information."""
    parser = argparse.ArgumentParser(description="Ollama Model Information for RAG")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    args = parser.parse_args()
    
    print("=== Ollama Model Recommendations for RAG ===\n")
    
    # Get installed models
    print("Checking for installed Ollama models...")
    installed_models = get_ollama_models(args.base_url)
    
    if installed_models:
        print(f"Found {len(installed_models)} installed models:\n")
        
        # Format model information
        model_data = []
        for model in installed_models:
            model_data.append([
                model.get("name"),
                model.get("modified", "Unknown"),
                f"{model.get('size', 0) / (1024*1024*1024):.2f} GB"
            ])
        
        # Display installed models
        print(tabulate(
            model_data,
            headers=["Model", "Last Modified", "Size"],
            tablefmt="grid"
        ))
        print()
    else:
        print("No installed models found or Ollama is not running.\n")
    
    # Display recommendations
    recommendations = recommend_models()
    
    print("=== Recommended Embedding Models ===\n")
    embed_data = []
    for model in recommendations["embedding"]["best"]:
        embed_data.append([
            model["name"],
            model["description"],
            model["size"],
            model["notes"]
        ])
    
    print(tabulate(
        embed_data,
        headers=["Model", "Description", "Size", "Notes"],
        tablefmt="grid"
    ))
    print()
    
    print("=== Recommended LLM Models ===\n")
    llm_data = []
    for model in recommendations["llm"]["best"]:
        llm_data.append([
            model["name"],
            model["description"],
            model["size"],
            model["notes"]
        ])
    
    print(tabulate(
        llm_data,
        headers=["Model", "Description", "Size", "Notes"],
        tablefmt="grid"
    ))
    print()
    
    print("=== Alternative LLM Models (for resource-constrained environments) ===\n")
    alt_data = []
    for model in recommendations["llm"]["alternatives"]:
        alt_data.append([
            model["name"],
            model["description"],
            model["size"],
            model["notes"]
        ])
    
    print(tabulate(
        alt_data,
        headers=["Model", "Description", "Size", "Notes"],
        tablefmt="grid"
    ))
    print()
    
    # Installation instructions
    print("=== Installation Instructions ===\n")
    print("To install a model, run:")
    print("  ollama pull MODEL_NAME\n")
    print("Example:")
    print("  ollama pull llama3")
    print("  ollama pull nomic-embed-text\n")
    
    # Configuration instructions
    print("=== Configuration Instructions ===\n")
    print("To use these models with the RAG pipeline, update your .env file:")
    print("  OLLAMA_EMBED_MODEL=nomic-embed-text")
    print("  OLLAMA_LLM_MODEL=llama3\n")


if __name__ == "__main__":
    main() 