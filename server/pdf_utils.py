# server/pdf_utils.py - Updated to use PyPDF2 instead of PyMuPDF
import requests
import tempfile
import os
import io
from PyPDF2 import PdfReader
from typing import Optional


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from local PDF file using PyPDF2"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            full_text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            return full_text.strip()
            
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Download PDF from URL and extract text using PyPDF2"""
    try:
        # Download PDF with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, timeout=30, headers=headers, stream=True)
        response.raise_for_status()
        
        # Method 1: Direct in-memory processing (preferred)
        try:
            pdf_content = response.content
            reader = PdfReader(io.BytesIO(pdf_content))
            full_text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            return full_text.strip()
            
        except Exception as memory_error:
            print(f"In-memory processing failed: {memory_error}, trying file method...")
            
            # Method 2: Fallback to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract text using file method
                text = extract_text_from_pdf(tmp_file_path)
                return text
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
    except Exception as e:
        print(f"Error downloading and extracting PDF from URL: {e}")
        return ""


def extract_text_from_pdf_content(pdf_content: bytes) -> str:
    """Extract text from PDF content bytes (useful for file uploads)"""
    try:
        reader = PdfReader(io.BytesIO(pdf_content))
        full_text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        return full_text.strip()
        
    except Exception as e:
        print(f"Error extracting text from PDF content: {e}")
        return ""


# Optional: Fallback function for compatibility
def safe_extract_text_from_pdf_url(pdf_url: str, max_retries: int = 2) -> str:
    """Safe PDF extraction with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            result = extract_text_from_pdf_url(pdf_url)
            if result.strip():  # If we got text, return it
                return result
            print(f"Attempt {attempt + 1}: No text extracted from PDF")
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries:
                print("All extraction attempts failed")
                return ""
    
    return ""
