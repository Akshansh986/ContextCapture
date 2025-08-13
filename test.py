import subprocess
import pyautogui
import time
from PIL import Image
import datetime
import os
import requests
import json
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
import signal
import sys
import argparse
import boto3

# Load environment variables from .env file
load_dotenv()

# Create directories if they don't exist
SCREENSHOT_DIR = "screenshots"
ACTIVITY_LOG_DIR = "activity_logs"

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)
    
if not os.path.exists(ACTIVITY_LOG_DIR):
    os.makedirs(ACTIVITY_LOG_DIR)

# Global variables for power state management
is_system_awake = True
power_monitor_running = True
power_state_lock = threading.Lock()

# Global variables for model configuration
MODEL_TYPE = "ollama"  # Default to ollama
AWS_BEDROCK_API_KEY = None
AWS_REGION = "us-east-1"  # Default region


def run_tesseract_ocr(image_path):
    """
    Run Tesseract OCR on an image and return the extracted text in plain text format
    """
    try:
        # Run tesseract command with plain text output format
        result = subprocess.run(['tesseract', image_path, 'stdout'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Tesseract OCR failed: {e}")
        return f"OCR Error: {e}"
    except FileNotFoundError:
        return "Error: Tesseract not found. Please install tesseract."

def analyze_with_claude(ocr_text):
    """
    Send OCR text to Claude via AWS Bedrock to analyze what the user is doing
    """
    try:
        # Set the API key as environment variable (as recommended by AWS)
        if AWS_BEDROCK_API_KEY:
            os.environ['AWS_BEARER_TOKEN_BEDROCK'] = AWS_BEDROCK_API_KEY
        
        # Create Bedrock runtime client using API key authentication
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION
        )
        
        prompt = f"""Based on the following text extracted from a user's screen, give 1 line description of what user is doing.

OCR Text:
{ocr_text}"""

        # Use the Converse API as recommended by AWS
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
        
        # Invoke the model using Converse API
        response = bedrock_runtime.converse(
            modelId="us.anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages
        )
        
        # Parse response from Converse API
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
        # Ollama API endpoint (default local installation)
        ollama_url = "http://localhost:11434/api/generate"
        
        # Create a prompt for the local ChatGPT model
        prompt = f"""Based on the following text extracted from a user's screen, give 1 line descripiton of  what user is doing.

OCR Text:
{ocr_text}
"""

        # Prepare the request payload for Ollama
        payload = {
            "model": "gpt-oss:20b",  # This should match your Ollama model name
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 1.0,
                "num_predict": 8000
            }
        }
        
        # Make HTTP request to Ollama
        response = requests.post(
            ollama_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            # Debug: Print the full response to understand the structure
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

def analyze_content(ocr_text):
    """
    Unified function to analyze OCR text using either Ollama or Claude (via AWS Bedrock) based on MODEL_TYPE
    """
    if MODEL_TYPE == "claude":
        if not AWS_BEDROCK_API_KEY and not os.getenv('AWS_BEARER_TOKEN_BEDROCK'):
            return "Error: AWS Bedrock API key not provided. Use --aws-bedrock-api-key or set AWS_BEARER_TOKEN_BEDROCK environment variable."
        return analyze_with_claude(ocr_text)
    else:  # Default to ollama
        return analyze_with_chatgpt(ocr_text)

def monitor_power_state():
    """
    Monitor macOS power state changes (sleep/wake) in a separate thread
    """
    global is_system_awake, power_monitor_running
    
    print("🔋 Starting power state monitor...")
    
    while power_monitor_running:
        try:
            # Use pmset to check if system is awake
            # pmset -g ps shows power source info
            result = subprocess.run(['pmset', '-g', 'ps'], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Check if system is in sleep state
                output = result.stdout.lower()
                
                # Simple heuristic: if we can run pmset successfully, system is likely awake
                # More sophisticated detection could parse the output for specific sleep states
                with power_state_lock:
                    if not is_system_awake:
                        is_system_awake = True
                        print("💡 System woke up - resuming monitoring")
            else:
                # If pmset fails, system might be sleeping or there's an issue
                with power_state_lock:
                    if is_system_awake:
                        is_system_awake = False
                        print("😴 System appears to be sleeping - pausing monitoring")
                        
        except subprocess.TimeoutExpired:
            # Timeout suggests system might be sleeping
            with power_state_lock:
                if is_system_awake:
                    is_system_awake = False
                    print("😴 System timeout detected - pausing monitoring")
        except Exception as e:
            print(f"Power monitor error: {e}")
            
        # Check every 10 seconds
        time.sleep(10)

def monitor_system_events():
    """
    Alternative method: Monitor macOS system events for sleep/wake
    """
    global is_system_awake, power_monitor_running
    
    try:
        # Monitor system log for sleep/wake events
        cmd = ['log', 'stream', '--predicate', 'subsystem == "com.apple.kernel" AND category == "PM"', '--level', 'info']
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        print("🔋 Monitoring system power events...")
        
        for line in iter(process.stdout.readline, ''):
            if not power_monitor_running:
                break
                
            line = line.strip().lower()
            
            if 'sleep' in line or 'going to sleep' in line:
                with power_state_lock:
                    if is_system_awake:
                        is_system_awake = False
                        print("😴 System going to sleep - pausing monitoring")
            elif 'wake' in line or 'waking up' in line or 'wake from' in line:
                with power_state_lock:
                    if not is_system_awake:
                        is_system_awake = True
                        print("💡 System waking up - resuming monitoring")
                        
    except Exception as e:
        print(f"System event monitor error: {e}")
        # Fallback to simple power state monitoring
        monitor_power_state()

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

def capture_and_analyze():
    # Take screenshot
    screenshot = pyautogui.screenshot()

    # Get current epoch timestamp
    epoch_time = int(time.time())
    timestamp_readable = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Resize image to have width of 1512px while maintaining aspect ratio
    width, height = screenshot.size
    new_width = 1512
    new_height = int(height * (new_width / width))
    resized_img = screenshot.resize((new_width, new_height))

    # Save only the resized image
    resized_filename = f"{epoch_time}.png"
    resized_path = os.path.join(SCREENSHOT_DIR, resized_filename)
    resized_img.save(resized_path)
    print(f"Resized screenshot saved to {resized_path} with dimensions {new_width}x{new_height}")

    # Process the screenshot through Tesseract OCR
    ocr_text = run_tesseract_ocr(resized_path)
    
    # Analyze OCR text with selected model (Ollama or Claude)
    model_name = "Claude API" if MODEL_TYPE == "claude" else "local ChatGPT model via Ollama"
    print(f"Analyzing content with {model_name}...")
    chatgpt_analysis = analyze_content(ocr_text)
    
    # Format the raw OCR log entry
    raw_log_entry = f"""
===== {timestamp_readable} (Epoch: {epoch_time}) =====
Screenshot: {resized_filename}
OCR Text:
{ocr_text}

"""
    
    # Format the processed analysis log entry
    analysis_header = f"{model_name} Analysis" if MODEL_TYPE == "claude" else "Local ChatGPT Analysis"
    processed_log_entry = f"""
===== {timestamp_readable} (Epoch: {epoch_time}) =====
Screenshot: {resized_filename}
{analysis_header} - What the user is doing:
{chatgpt_analysis}

"""
    
    # Save to raw activity log file
    raw_log_path = os.path.join(ACTIVITY_LOG_DIR, "activity_log_raw.txt")
    with open(raw_log_path, "a") as raw_log:
        raw_log.write(raw_log_entry)
    
    # Save to processed activity log file
    processed_log_path = os.path.join(ACTIVITY_LOG_DIR, "activity_log_processed.txt")
    with open(processed_log_path, "a") as processed_log:
        processed_log.write(processed_log_entry)
    
    print(f"Activity logged at {timestamp_readable}")
    print(f"{analysis_header}: {chatgpt_analysis}")
    
    return {"raw": raw_log_entry, "processed": processed_log_entry}

def parse_arguments():
    """
    Parse command line arguments for model selection and AWS Bedrock API key
    """
    parser = argparse.ArgumentParser(description='Screen monitoring with AI analysis')
    parser.add_argument('--model', choices=['ollama', 'claude'], default='ollama',
                       help='Choose AI model: ollama (local) or claude (via AWS Bedrock)')
    parser.add_argument('--aws-bedrock-api-key', type=str,
                       help='AWS Bedrock API key (or set AWS_BEARER_TOKEN_BEDROCK environment variable)')
    parser.add_argument('--aws-region', type=str, default='us-east-1',
                       help='AWS region for Bedrock (default: us-east-1)')
    parser.add_argument('--test', action='store_true',
                       help='Test the selected model connection and exit')
    
    return parser.parse_args()

# Main loop to run every 60 seconds
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set global model configuration
    MODEL_TYPE = args.model
    AWS_BEDROCK_API_KEY = args.aws_bedrock_api_key or os.getenv('AWS_BEARER_TOKEN_BEDROCK')
    AWS_REGION = args.aws_region
    
    # Check if user wants to test connection first
    if args.test:
        if MODEL_TYPE == "claude":
            if not AWS_BEDROCK_API_KEY:
                print("❌ AWS Bedrock API key required for testing. Use --aws-bedrock-api-key or set AWS_BEARER_TOKEN_BEDROCK environment variable.")
                sys.exit(1)
            print("Testing Claude via AWS Bedrock connection...")
            test_text = "Hello, please respond with 'Claude via Bedrock is working correctly!'"
            result = analyze_with_claude(test_text)
            if "working correctly" in result.lower() or "hello" in result.lower():
                print("✅ Claude via AWS Bedrock test successful!")
                print(f"Response: {result}")
            else:
                print("❌ Claude via AWS Bedrock test failed.")
                print(f"Response: {result}")
        else:  # ollama
            print("Testing Ollama connection...")
            if test_ollama_connection():
                print("✅ Ollama test successful!")
            else:
                print("❌ Ollama test failed. Please check your setup.")
        sys.exit(0)
    
    try:
        model_display = "Claude via AWS Bedrock" if MODEL_TYPE == "claude" else "local ChatGPT model via Ollama"
        print(f"🖥️  Starting intelligent screen monitoring with {model_display}")
        print("💤 Features: Automatic sleep/wake detection - monitoring pauses when laptop sleeps")
        print("🛑 Press Ctrl+C to stop")
        
        if MODEL_TYPE == "claude":
            if not AWS_BEDROCK_API_KEY:
                print("❌ AWS Bedrock API key required. Use --aws-bedrock-api-key or set AWS_BEARER_TOKEN_BEDROCK environment variable.")
                sys.exit(1)
            print("📝 Note: Using Claude via AWS Bedrock for analysis")
            print("🧪 Tip: Run 'python test.py --model claude --test' to test your AWS Bedrock connection first")
        else:
            print("📝 Note: Make sure Ollama is running with the 'gpt-oss:20b' model loaded on localhost:11434")
            print("🧪 Tip: Run 'python test.py --test' to test your Ollama connection first")
        
        # Test connection before starting
        print(f"\nTesting {model_display} connection before starting...")
        if MODEL_TYPE == "claude":
            test_text = "Hello, please respond briefly that you're working."
            result = analyze_with_claude(test_text)
            if "error" in result.lower():
                print("❌ Claude via AWS Bedrock connection test failed. Please check your AWS Bedrock API key and setup before continuing.")
                print(f"Error: {result}")
                sys.exit(1)
            print("✅ Claude via AWS Bedrock connection successful! Starting monitoring...\n")
        else:
            if not test_ollama_connection():
                print("❌ Ollama connection test failed. Please check your setup before continuing.")
                print("Make sure:")
                print("1. Ollama is running: ollama serve")
                print("2. Model is available: ollama list")
                print("3. Model name 'gpt-oss:20b' is correct")
                sys.exit(1)
            print("✅ Ollama connection successful! Starting monitoring...\n")
        
        # Start power monitoring in a separate thread
        def signal_handler(sig, frame):
            global power_monitor_running
            power_monitor_running = False
            print("\n🛑 Shutting down power monitor...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start the power monitoring thread
        power_thread = threading.Thread(target=monitor_system_events, daemon=True)
        power_thread.start()
        
        print("🔄 Starting main monitoring loop...")
        
        loop_count = 0
        while True:
            # Check if system is awake before taking screenshot
            with power_state_lock:
                system_awake = is_system_awake
            
            if system_awake:
                try:
                    result = capture_and_analyze()
                    print("Waiting 30 seconds before next capture...")
                except Exception as e:
                    print(f"Error during capture and analysis: {e}")
            else:
                print("😴 System sleeping - skipping capture...")
                
            # Show status every 5 minutes
            loop_count += 1
            if loop_count % 10 == 0:  # Every 10 loops (5 minutes at 30s intervals)
                status = "💡 AWAKE" if system_awake else "😴 SLEEPING"
                print(f"📊 Status update: {status} | Loop: {loop_count}")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        power_monitor_running = False
        print("\n🛑 Monitoring stopped by user.")