#!/usr/bin/env python3
"""
Script to check if Ollama is installed and running.
"""

import os
import sys
import subprocess
import requests
import platform

def check_ollama_installed():
    """Check if Ollama is installed."""
    system = platform.system().lower()
    
    if system == "darwin" or system == "linux":
        try:
            result = subprocess.run(
                ["which", "ollama"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
    elif system == "windows":
        try:
            result = subprocess.run(
                ["where", "ollama"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
    return False

def check_ollama_running(base_url="http://localhost:11434"):
    """Check if Ollama is running."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def main():
    """Main function."""
    print("=== Ollama Status Check ===\n")
    
    # Check if Ollama is installed
    if check_ollama_installed():
        print("✅ Ollama is installed")
    else:
        print("❌ Ollama is not installed")
        print("\nTo install Ollama, visit: https://ollama.ai/download")
        print("Follow the installation instructions for your platform.")
        sys.exit(1)
    
    # Check if Ollama is running
    if check_ollama_running():
        print("✅ Ollama is running")
        print("\nYou're all set! You can now run the RAG pipeline.")
    else:
        print("❌ Ollama is not running")
        
        system = platform.system().lower()
        if system == "darwin" or system == "linux":
            print("\nTo start Ollama, open a new terminal and run:")
            print("  ollama serve")
        elif system == "windows":
            print("\nTo start Ollama, open a new command prompt and run:")
            print("  ollama serve")
        else:
            print("\nPlease start the Ollama service according to your platform's instructions.")
        
        print("\nAfter starting Ollama, you can run this check again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 