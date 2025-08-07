# server/chunker.py
from typing import List

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move start position
        if end >= text_length:
            break
        start += chunk_size - overlap
    
    return chunks

def smart_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Advanced chunking that tries to break at sentence boundaries
    """
    if not text:
        return []
    
    # First, split by paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Paragraph itself is too long, use basic chunking
                if len(paragraph) > chunk_size:
                    sub_chunks = chunk_text(paragraph, chunk_size, overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
