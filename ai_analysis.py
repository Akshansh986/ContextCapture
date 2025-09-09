import os
import requests
import json
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential


def analyze_with_claude(ocr_text, aws_bedrock_api_key=None, aws_region="us-east-1"):
    """
    Send OCR text to Claude via AWS Bedrock to analyze what the user is doing
    """
    try:
        if aws_bedrock_api_key:
            os.environ['AWS_BEARER_TOKEN_BEDROCK'] = aws_bedrock_api_key
        
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region
        )
        
        prompt = f"""Based on the following text extracted from a user's screen, give 1 line description of what user is doing.

OCR Text:
{ocr_text}"""

        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
        
        response = bedrock_runtime.converse(
            modelId="us.anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages
        )
        
        if "output" in response and "message" in response["output"]:
            content = response["output"]["message"]["content"]
            if content and len(content) > 0:
                return content[0]["text"].strip()
        
        return "Error: No content in Claude Bedrock response"
            
    except Exception as e:
        return f"Analysis Error: AWS Bedrock request failed: {str(e)}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_with_chatgpt(ocr_text):
    """
    Send OCR text to local Ollama ChatGPT OSS 20B model to analyze what the user is doing
    """
    try:
        ollama_url = "http://localhost:11434/api/generate"
        
        prompt = f"""Based on the following text extracted from a user's screen, give 1 line descripiton of  what user is doing.

OCR Text:
{ocr_text}
"""

        payload = {
            "model": "gpt-oss:20b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 1.0,
                "num_predict": 8000
            }
        }
        
        response = requests.post(
            ollama_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"DEBUG: Full Ollama response: {result}")
            
            if "response" in result:
                model_output = result["response"].strip()
                print(f"DEBUG: Model output length: {len(model_output)}")
                print(f"DEBUG: Model output: '{model_output}'")
                
                if not model_output:
                    return "Error: Model returned empty response"
                return model_output
            else:
                return f"Error: No response field in Ollama output. Available fields: {list(result.keys())}"
        else:
            return f"Error: Ollama API returned status code {response.status_code}: {response.text}"
        
    except requests.exceptions.ConnectionError:
        return "Analysis Error: Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
    except requests.exceptions.Timeout:
        return "Analysis Error: Request to Ollama timed out"
    except requests.exceptions.RequestException as e:
        return f"Analysis Error: Request failed: {str(e)}"
    except json.JSONDecodeError:
        return "Analysis Error: Invalid JSON response from Ollama"
    except Exception as e:
        return f"Analysis Error: {str(e)}"


def analyze_content(ocr_text, model_type="ollama", aws_bedrock_api_key=None, aws_region="us-east-1"):
    """
    Unified function to analyze OCR text using either Ollama or Claude (via AWS Bedrock) based on model_type
    """
    if model_type == "claude":
        if not aws_bedrock_api_key and not os.getenv('AWS_BEARER_TOKEN_BEDROCK'):
            return "Error: AWS Bedrock API key not provided. Use --aws-bedrock-api-key or set AWS_BEARER_TOKEN_BEDROCK environment variable."
        return analyze_with_claude(ocr_text, aws_bedrock_api_key, aws_region)
    else:
        return analyze_with_chatgpt(ocr_text)


def test_ollama_connection():
    """
    Test function to verify Ollama is working with a simple prompt
    """
    try:
        ollama_url = "http://localhost:11434/api/generate"
        test_payload = {
            "model": "gpt-oss:20b",
            "prompt": "Hello, please respond with 'Ollama is working correctly!'",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200
            }
        }
        
        print("Testing Ollama connection...")
        response = requests.post(
            ollama_url,
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Full response: {result}")
            if "response" in result:
                print(f"Model response: '{result['response']}'")
                return True
            else:
                print(f"No 'response' field. Available fields: {list(result.keys())}")
                return False
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False
