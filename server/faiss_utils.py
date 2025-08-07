# server/faiss_utils.py - Railway compatible version without FAISS
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

def create_faiss_index(embeddings: List[List[float]]) -> Dict[str, Any]:
    """Create a simple index using numpy arrays (Railway compatible)"""
    try:
        if not embeddings:
            return None
            
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        return {
            "embeddings": embeddings_array,
            "dimension": embeddings_array.shape[1],
            "size": len(embeddings_array)
        }
        
    except Exception as e:
        print(f"Error creating index: {e}")
        return None

def search_similar_chunks(
    query_embedding: List[float], 
    index: Dict[str, Any], 
    chunks: List[str], 
    k: int = 5
) -> List[str]:
    """Search for similar chunks using cosine similarity (Railway compatible)"""
    try:
        if index is None or not chunks or index["size"] == 0:
            return []
            
        query_array = np.array([query_embedding], dtype=np.float32)
        embeddings = index["embeddings"]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_array, embeddings)[0]
        
        # Get top k most similar indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return actual text chunks
        similar_chunks = []
        for idx in top_indices:
            if 0 <= idx < len(chunks) and similarities[idx] > 0.3:  # Similarity threshold
                similar_chunks.append(chunks[idx])
        
        return similar_chunks
        
    except Exception as e:
        print(f"Error searching similar chunks: {e}")
        return []
