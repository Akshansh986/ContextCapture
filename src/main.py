import time
import datetime
import os
import signal
import sys
import argparse
import threading
from dotenv import load_dotenv

from ocr_service import run_tesseract_ocr
from ai_analysis import analyze_content
from storage_service import ensure_directories, save_screenshot, write_activity_logs
from screenshot_service import capture_screenshot, resize_screenshot
from test_connection import test_connection_before_start
import power_monitor

# Load environment variables from .env file
load_dotenv()

# Ensure directories exist
ensure_directories()

# Global variables for model configuration
MODEL_TYPE = "ollama"  # Default to ollama


def capture_and_analyze():
    screenshot, epoch_time = capture_screenshot()
    timestamp_readable = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    resized_img, new_width, new_height = resize_screenshot(screenshot)

    resized_filename, resized_path = save_screenshot(resized_img, epoch_time)
    print(f"Resized screenshot saved to {resized_path} with dimensions {new_width}x{new_height}")

    ocr_text = run_tesseract_ocr(resized_path)
    
    model_name = "Claude API" if MODEL_TYPE == "claude" else "local ChatGPT model via Ollama"
    print(f"Analyzing content with {model_name}...")
    analysis = analyze_content(ocr_text, MODEL_TYPE)
    
    log_entries = write_activity_logs(epoch_time, resized_filename, ocr_text, analysis, MODEL_TYPE)
    
    analysis_header = f"{model_name} Analysis" if MODEL_TYPE == "claude" else "Local ChatGPT Analysis"
    print(f"Activity logged at {timestamp_readable}")
    print(f"{analysis_header}: {analysis}")
    
    return log_entries

def parse_arguments():
    """
    Parse command line arguments for model selection
    """
    parser = argparse.ArgumentParser(description='Screen monitoring with AI analysis')
    parser.add_argument('--model', choices=['ollama', 'claude'], default='ollama',
                       help='Choose AI model: ollama (local) or claude (via AWS Bedrock)')
    
    return parser.parse_args()

# Main loop to run every 60 seconds
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set global model configuration
    MODEL_TYPE = args.model
    
    try:
        print(f"🖥️  Starting intelligent screen monitoring")
        print("🛑 Press Ctrl+C to stop")
        
        # Test connection before starting
        if not test_connection_before_start(MODEL_TYPE):
            sys.exit(1)
        
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