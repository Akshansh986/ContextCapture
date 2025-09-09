import os
from ai_analyzer import ClaudeAnalyzer, OpenAIAnalyzer


def analyze_content(ocr_text, model_type="ollama", aws_bedrock_api_key=None, aws_region="us-east-1"):
    """
    Unified function to analyze OCR text using either Ollama or Claude (via AWS Bedrock) based on model_type
    """
    if model_type == "claude":
        if not aws_bedrock_api_key and not os.getenv('AWS_BEARER_TOKEN_BEDROCK'):
            return "Error: AWS Bedrock API key not provided. Use --aws-bedrock-api-key or set AWS_BEARER_TOKEN_BEDROCK environment variable."
        analyzer = ClaudeAnalyzer(aws_bedrock_api_key, aws_region)
        return analyzer.analyze(ocr_text)
    else:
        analyzer = OpenAIAnalyzer()
        return analyzer.analyze(ocr_text)


def test_ollama_connection():
    """
    Test function to verify Ollama is working with a simple prompt
    """
    analyzer = OpenAIAnalyzer()
    return analyzer.test_connection()


def analyze_with_claude(ocr_text, aws_bedrock_api_key=None, aws_region="us-east-1"):
    """
    Legacy function for backward compatibility
    """
    analyzer = ClaudeAnalyzer(aws_bedrock_api_key, aws_region)
    return analyzer.analyze(ocr_text)
