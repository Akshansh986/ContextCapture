import sys
import os
from ai_analyzer import ClaudeAnalyzer, OpenAIAnalyzer


def test_claude_connection(aws_bedrock_api_key, aws_region):
    if not aws_bedrock_api_key:
        print("❌ AWS Bedrock API key required for testing. Use --aws-bedrock-api-key or set AWS_BEARER_TOKEN_BEDROCK environment variable.")
        return False
    
    print("Testing Claude via AWS Bedrock connection...")
    analyzer = ClaudeAnalyzer(aws_bedrock_api_key, aws_region)
    test_text = "Hello, please respond with 'Claude via Bedrock is working correctly!'"
    result = analyzer.analyze(test_text)
    
    if "working correctly" in result.lower() or "hello" in result.lower():
        print("✅ Claude via AWS Bedrock test successful!")
        print(f"Response: {result}")
        return True
    else:
        print("❌ Claude via AWS Bedrock test failed.")
        print(f"Response: {result}")
        return False


def test_ollama_connection_wrapper():
    print("Testing Ollama connection...")
    analyzer = OpenAIAnalyzer()
    if analyzer.test_connection():
        print("✅ Ollama test successful!")
        return True
    else:
        print("❌ Ollama test failed. Please check your setup.")
        return False


def run_connection_test(model_type, aws_bedrock_api_key=None, aws_region="us-east-1"):
    if model_type == "claude":
        return test_claude_connection(aws_bedrock_api_key, aws_region)
    else:  # ollama
        return test_ollama_connection_wrapper()


def test_connection_before_start(model_type, aws_bedrock_api_key=None, aws_region="us-east-1"):
    model_display = "Claude via AWS Bedrock" if model_type == "claude" else "local ChatGPT model via Ollama"
    print(f"\nTesting {model_display} connection before starting...")
    
    if model_type == "claude":
        analyzer = ClaudeAnalyzer(aws_bedrock_api_key, aws_region)
        test_text = "Hello, please respond briefly that you're working."
        result = analyzer.analyze(test_text)
        if "error" in result.lower():
            print("❌ Claude via AWS Bedrock connection test failed. Please check your AWS Bedrock API key and setup before continuing.")
            print(f"Error: {result}")
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
