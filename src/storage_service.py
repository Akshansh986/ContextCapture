import os
import datetime


SCREENSHOT_DIR = "../screenshots"
ACTIVITY_LOG_DIR = "../activity_logs"


def ensure_directories():
    """
    Create directories if they don't exist
    """
    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)
        
    if not os.path.exists(ACTIVITY_LOG_DIR):
        os.makedirs(ACTIVITY_LOG_DIR)


def save_screenshot(screenshot, epoch_time):
    """
    Save screenshot to the screenshots directory
    """
    filename = f"{epoch_time}.png"
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    screenshot.save(filepath)
    return filename, filepath


def write_activity_logs(epoch_time, screenshot_filename, ocr_text, analysis, model_type):
    """
    Write both raw and processed activity logs
    """
    timestamp_readable = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    raw_log_entry = f"""
===== {timestamp_readable} (Epoch: {epoch_time}) =====
Screenshot: {screenshot_filename}
OCR Text:
{ocr_text}

"""
    
    model_name = "Bedrock API" if model_type == "bedrock" else "local ChatGPT model via Ollama"
    analysis_header = f"{model_name} Analysis" if model_type == "bedrock" else "Local ChatGPT Analysis"
    processed_log_entry = f"""
===== {timestamp_readable} (Epoch: {epoch_time}) =====
Screenshot: {screenshot_filename}
{analysis_header} - What the user is doing:
{analysis}

"""
    
    raw_log_path = os.path.join(ACTIVITY_LOG_DIR, "activity_log_raw.txt")
    with open(raw_log_path, "a") as raw_log:
        raw_log.write(raw_log_entry)
    
    processed_log_path = os.path.join(ACTIVITY_LOG_DIR, "activity_log_processed.txt")
    with open(processed_log_path, "a") as processed_log:
        processed_log.write(processed_log_entry)
    
    return {"raw": raw_log_entry, "processed": processed_log_entry}
