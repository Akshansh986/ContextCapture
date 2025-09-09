import requests
import json
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIAnalyzer:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model_name="gpt-oss:20b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze(self, ocr_text):
        """
        Send OCR text to local Ollama ChatGPT OSS 20B model to analyze what the user is doing
        """
        try:
            prompt = f"""Based on the following text extracted from a user's screen, give 1 line descripiton of  what user is doing.

OCR Text:
{ocr_text}
"""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "num_predict": 8000
                }
            }
            
            response = requests.post(
                self.ollama_url,
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
    
    def test_connection(self):
        """
        Test function to verify Ollama is working with a simple prompt
        """
        try:
            test_payload = {
                "model": self.model_name,
                "prompt": "Hello, please respond with 'Ollama is working correctly!'",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200
                }
            }
            
            print("Testing Ollama connection...")
            response = requests.post(
                self.ollama_url,
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
