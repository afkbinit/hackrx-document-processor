# server/pdf_utils.py - Original version with PyMuPDF
import fitz  # PyMuPDF
import requests
import tempfile
import os
from typing import Optional


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from local PDF file"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Download PDF from URL and extract text"""
    try:
        # Download PDF
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        try:
            # Extract text
            text = extract_text_from_pdf(tmp_file_path)
            return text
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        print(f"Error downloading and extracting PDF from URL: {e}")
        return ""
