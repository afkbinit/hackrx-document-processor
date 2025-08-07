# server/faiss_utils.py - Simple vector similarity without FAISS
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


def create_faiss_index(embeddings: List[List[float]]) -> Dict[str, Any]:
    """Create a simple index using numpy arrays instead of FAISS"""
    try:
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        return {
            "embeddings": embeddings_array,
            "dimension": embeddings_array.shape[1] if len(embeddings_array) > 0 else 0,
            "size": len(embeddings_array)
        }
    except Exception as e:
        print(f"Error creating index: {e}")
        return {"embeddings": np.array([]), "dimension": 0, "size": 0}


def search_similar_chunks(
    index: Dict[str, Any], 
    query_embedding: List[float], 
    k: int = 5
) -> List[Tuple[float, int]]:
    """Search for similar chunks using cosine similarity instead of FAISS"""
    try:
        if index["size"] == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        embeddings = index["embeddings"]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_array, embeddings)[0]
        
        # Get top k most similar indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return as (similarity_score, index) tuples
        results = [(float(similarities[idx]), int(idx)) for idx in top_indices]
        
        return results
        
    except Exception as e:
        print(f"Error searching similar chunks: {e}")
        return []


# Compatibility functions for existing code
def add_to_index(index: Dict[str, Any], new_embeddings: List[List[float]]) -> Dict[str, Any]:
    """Add new embeddings to existing index"""
    try:
        if index["size"] == 0:
            return create_faiss_index(new_embeddings)
        
        existing_embeddings = index["embeddings"]
        new_embeddings_array = np.array(new_embeddings, dtype=np.float32)
        
        combined_embeddings = np.vstack([existing_embeddings, new_embeddings_array])
        
        return {
            "embeddings": combined_embeddings,
            "dimension": combined_embeddings.shape[1],
            "size": len(combined_embeddings)
        }
    except Exception as e:
        print(f"Error adding to index: {e}")
        return index
