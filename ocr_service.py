import subprocess


def run_tesseract_ocr(image_path):
    """
    Run Tesseract OCR on an image and return the extracted text in plain text format
    """
    try:
        result = subprocess.run(['tesseract', image_path, 'stdout'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Tesseract OCR failed: {e}")
        return f"OCR Error: {e}"
    except FileNotFoundError:
        return "Error: Tesseract not found. Please install tesseract."
