import os
from ai_analyzer import BedrockAnalyzer, OpenAIAnalyzer


def analyze_content(ocr_text, model_type="ollama"):
    """
    Unified function to analyze OCR text using either Ollama or Bedrock based on model_type
    """
    if model_type == "bedrock":
        analyzer = BedrockAnalyzer()
        return analyzer.analyze(ocr_text)
    else:
        analyzer = OpenAIAnalyzer()
        return analyzer.analyze(ocr_text)