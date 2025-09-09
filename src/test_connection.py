import sys
import os
from ai_analyzer import BedrockAnalyzer, OpenAIAnalyzer

def test_connection_before_start(model_type):
    """
    Test connection before starting the main monitoring loop
    """
    model_display = "Bedrock" if model_type == "bedrock" else "local ChatGPT model via Ollama"
    print(f"\nTesting {model_display} connection before starting...")
    
    if model_type == "bedrock":
        analyzer = BedrockAnalyzer()
        if not analyzer.test_connection():
            return False
        print("✅ Bedrock connection successful! Starting monitoring...\n")
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
