# server/pdf_utils.py - PyPDF2 version (Railway compatible)
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
        
        # Direct in-memory processing
        pdf_content = response.content
        reader = PdfReader(io.BytesIO(pdf_content))
        full_text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        return full_text.strip()
            
    except Exception as e:
        print(f"Error downloading and extracting PDF from URL: {e}")
        return ""
