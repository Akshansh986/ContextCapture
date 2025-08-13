## ContextCapture

**One-line**: A macOS tool that periodically screenshots your screen, OCRs the text, and uses a local (Ollama) or cloud (Claude via Bedrock) AI model to log a one-line summary of your activity.

### Prerequisites
- **macOS** with Screen Recording and Accessibility permissions available
- **Python 3.10+**
- **Homebrew** (`/bin/zsh` recommended)
- **Tesseract OCR**: `brew install tesseract`
- Optional: **Ollama** for local model (`gpt-oss:20b`) or **AWS Bedrock** access for Claude

### Setup
```bash
git clone <this-repo>
cd ContextCapture

# Create and activate venv (required before running any Python script)
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Verify Tesseract is installed
tesseract --version
```

### macOS permissions
Grant permissions in System Settings → Privacy & Security:
- **Screen Recording**: add Terminal or your IDE
- **Accessibility**: add Terminal and Python

### Option A: Run with local model (Ollama)
```bash
# Install Ollama from https://ollama.com and start the server
ollama serve

# Pull the model used by this project
ollama pull gpt-oss:20b

# From the project root, with venv active
python test.py --test            # quick connection test
python test.py                   # starts monitoring using Ollama (default)
```

### Option B: Run with Claude via AWS Bedrock
```bash
# Set your Bedrock bearer token or pass via flag
export AWS_BEARER_TOKEN_BEDROCK="<your_bedrock_token>"

# Test and run
python test.py --model claude --test
python test.py --model claude

# You can also pass region and key explicitly
python test.py --model claude --aws-region us-east-1 --aws-bedrock-api-key <token>
```

### Outputs
- **Screenshots**: `screenshots/` (PNG, width 1512 px)
- **Logs**: `activity_logs/activity_log_raw.txt` (OCR text) and `activity_logs/activity_log_processed.txt` (one-line AI summary)

Note: The repo tracks empty `screenshots/` and `activity_logs/` via `.gitkeep` files, but ignores generated contents.

### Troubleshooting
- **"Error: Tesseract not found"**: `brew install tesseract`, then ensure `tesseract --version` works in the same shell
- **Black/empty screenshots or permission errors**: grant Screen Recording and Accessibility permissions, then restart Terminal/IDE
- **Ollama connection errors**: ensure `ollama serve` is running and `ollama pull gpt-oss:20b` completed
- **Bedrock errors**: ensure your token/region is correct and Bedrock access is enabled for your account

### Useful commands
```bash
# Always activate venv before running scripts
source venv/bin/activate

# Test Ollama
python test.py --test

# Test Claude via Bedrock
python test.py --model claude --test

# Run normally (default: Ollama)
python test.py
```

