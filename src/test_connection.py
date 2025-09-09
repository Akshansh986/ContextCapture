import sys
import os
from ai_analyzer import ClaudeAnalyzer, OpenAIAnalyzer

def test_connection_before_start(model_type):
    """
    Test connection before starting the main monitoring loop
    """
    model_display = "Claude via AWS Bedrock" if model_type == "claude" else "local ChatGPT model via Ollama"
    print(f"\nTesting {model_display} connection before starting...")
    
    if model_type == "claude":
        analyzer = ClaudeAnalyzer()
        if not analyzer.test_connection():
            return False
        print("✅ Claude via AWS Bedrock connection successful! Starting monitoring...\n")
        return True
    else:
        analyzer = OpenAIAnalyzer()
        if not analyzer.test_connection():
            print("❌ Ollama connection test failed. Please check your setup before continuing.")
            print("Make sure:")
            print("1. Ollama is running: ollama serve")
            print("2. Model is available: ollama list")
            print("3. Model name 'gpt-oss:20b' is correct")
            return False
        print("✅ Ollama connection successful! Starting monitoring...\n")
        return True
