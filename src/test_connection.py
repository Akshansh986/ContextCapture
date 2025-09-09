import sys
import os
from ai_analyzer import ClaudeAnalyzer, OpenAIAnalyzer


def run_connection_test(model_type, aws_bedrock_api_key=None, aws_region="us-east-1"):
    """
    Run connection test for the specified model type
    """
    if model_type == "claude":
        if not aws_bedrock_api_key:
            print("❌ AWS Bedrock API key required for testing. Use --aws-bedrock-api-key or set AWS_BEARER_TOKEN_BEDROCK environment variable.")
            return False
        
        analyzer = ClaudeAnalyzer(aws_bedrock_api_key, aws_region)
        print("Testing Claude via AWS Bedrock connection...")
        
        if analyzer.test_connection():
            print("✅ Claude via AWS Bedrock test successful!")
            return True
        else:
            print("❌ Claude via AWS Bedrock test failed.")
            return False
    else:  # ollama
        analyzer = OpenAIAnalyzer()
        return analyzer.test_connection()


def test_connection_before_start(model_type, aws_bedrock_api_key=None, aws_region="us-east-1"):
    """
    Test connection before starting the main monitoring loop
    """
    model_display = "Claude via AWS Bedrock" if model_type == "claude" else "local ChatGPT model via Ollama"
    print(f"\nTesting {model_display} connection before starting...")
    
    if model_type == "claude":
        if not aws_bedrock_api_key:
            print("❌ AWS Bedrock API key required. Please check your AWS Bedrock API key and setup before continuing.")
            return False
            
        analyzer = ClaudeAnalyzer(aws_bedrock_api_key, aws_region)
        if not analyzer.test_connection():
            print("❌ Claude via AWS Bedrock connection test failed. Please check your AWS Bedrock API key and setup before continuing.")
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
