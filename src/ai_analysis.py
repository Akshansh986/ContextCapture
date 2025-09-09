import os
from ai_analyzer import ClaudeAnalyzer, OpenAIAnalyzer


def analyze_content(ocr_text, model_type="ollama"):
    """
    Unified function to analyze OCR text using either Ollama or Claude (via AWS Bedrock) based on model_type
    """
    if model_type == "claude":
        analyzer = ClaudeAnalyzer()
        return analyzer.analyze(ocr_text)
    else:
        analyzer = OpenAIAnalyzer()
        return analyzer.analyze(ocr_text)