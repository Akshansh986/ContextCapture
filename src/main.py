import time
import datetime
import os
import signal
import sys
import argparse
import threading
from dotenv import load_dotenv

from ocr_service import run_tesseract_ocr
from ai_analysis import analyze_content, test_ollama_connection, analyze_with_claude
from storage_service import ensure_directories, save_screenshot, write_activity_logs
from screenshot_service import capture_screenshot, resize_screenshot
import power_monitor

# Load environment variables from .env file
load_dotenv()

# Ensure directories exist
ensure_directories()

# Global variables for model configuration
MODEL_TYPE = "ollama"  # Default to ollama
AWS_BEDROCK_API_KEY = None
AWS_REGION = "us-east-1"  # Default region







def capture_and_analyze():
    screenshot, epoch_time = capture_screenshot()
    timestamp_readable = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    resized_img, new_width, new_height = resize_screenshot(screenshot)

    resized_filename, resized_path = save_screenshot(resized_img, epoch_time)
    print(f"Resized screenshot saved to {resized_path} with dimensions {new_width}x{new_height}")

    ocr_text = run_tesseract_ocr(resized_path)
    
    model_name = "Claude API" if MODEL_TYPE == "claude" else "local ChatGPT model via Ollama"
    print(f"Analyzing content with {model_name}...")
    analysis = analyze_content(ocr_text, MODEL_TYPE, AWS_BEDROCK_API_KEY, AWS_REGION)
    
    log_entries = write_activity_logs(epoch_time, resized_filename, ocr_text, analysis, MODEL_TYPE)
    
    analysis_header = f"{model_name} Analysis" if MODEL_TYPE == "claude" else "Local ChatGPT Analysis"
    print(f"Activity logged at {timestamp_readable}")
    print(f"{analysis_header}: {analysis}")
    
    return log_entries

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
            result = analyze_with_claude(test_text, AWS_BEDROCK_API_KEY, AWS_REGION)
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
            result = analyze_with_claude(test_text, AWS_BEDROCK_API_KEY, AWS_REGION)
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
            power_monitor.stop_monitoring()
            print("\n🛑 Shutting down power monitor...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start the power monitoring thread
        power_thread = threading.Thread(target=power_monitor.monitor_system_events, daemon=True)
        power_thread.start()
        
        print("🔄 Starting main monitoring loop...")
        
        loop_count = 0
        while True:
            system_awake = power_monitor.is_awake()
            
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
        power_monitor.stop_monitoring()
        print("\n🛑 Monitoring stopped by user.")