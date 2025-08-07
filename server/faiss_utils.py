# server/faiss_utils.py - Original version with FAISS
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any


def create_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    """Create FAISS index for similarity search"""
    try:
        if not embeddings:
            return None
            
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]
        
        # Create index
        index = faiss.IndexFlatIP(dimension)  # Inner product index
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add embeddings to index
        index.add(embeddings_array)
        
        return index
        
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None


def search_similar_chunks(
    query_embedding: List[float], 
    index: faiss.Index, 
    chunks: List[str], 
    k: int = 5
) -> List[str]:
    """Search for similar chunks using FAISS"""
    try:
        if index is None or not chunks:
            return []
            
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = index.search(query_array, min(k, len(chunks)))
        
        # Return actual text chunks
        similar_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                similar_chunks.append(chunks[idx])
        
        return similar_chunks
        
    except Exception as e:
        print(f"Error searching similar chunks: {e}")
        return []
