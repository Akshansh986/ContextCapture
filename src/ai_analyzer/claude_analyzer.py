import os
import boto3


class ClaudeAnalyzer:
    def __init__(self, aws_region="us-east-1"):
        self.aws_region = aws_region
        
    def analyze(self, ocr_text):
        """
        Send OCR text to Claude via AWS Bedrock to analyze what the user is doing
        """
        try:
            aws_bedrock_api_key = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
            if not aws_bedrock_api_key:
                return "Error: AWS_BEARER_TOKEN_BEDROCK environment variable not set"
            
            bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.aws_region
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
    
    def test_connection(self):
        """
        Test function to verify Claude via AWS Bedrock is working
        """
        aws_bedrock_api_key = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        if not aws_bedrock_api_key:
            print("❌ AWS_BEARER_TOKEN_BEDROCK environment variable not set")
            return False
            
        test_text = "Hello, please respond with 'Claude via Bedrock is working correctly!'"
        result = self.analyze(test_text)
        
        if "error" in result.lower():
            print(f"❌ Claude connection test failed: {result}")
            return False
            
        success = "working correctly" in result.lower() or "hello" in result.lower()
        if success:
            print("✅ Claude via AWS Bedrock test successful!")
            print(f"Response: {result}")
        else:
            print("❌ Claude via AWS Bedrock test failed.")
            print(f"Response: {result}")
        return success
