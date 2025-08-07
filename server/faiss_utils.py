# server/faiss_utils.py - Reliable Railway-compatible version
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def create_faiss_index(embeddings: List[List[float]]) -> Dict[str, Any]:
    """Create numpy-based index for similarity search"""
    try:
        if not embeddings or not any(embeddings):
            logger.error("No valid embeddings provided")
            return None
            
        # Filter out empty embeddings
        valid_embeddings = [emb for emb in embeddings if emb and len(emb) > 0]
        
        if not valid_embeddings:
            logger.error("No valid embeddings after filtering")
            return None
        
        embeddings_array = np.array(valid_embeddings, dtype=np.float32)
        
        logger.info(f"Created index with {len(valid_embeddings)} embeddings")
        
        return {
            "embeddings": embeddings_array,
            "dimension": embeddings_array.shape[1],
            "size": len(valid_embeddings)
        }
        
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return None

def search_similar_chunks(
    query_embedding: List[float], 
    index: Dict[str, Any], 
    chunks: List[str], 
    k: int = 5
) -> List[str]:
    """Search for similar chunks using cosine similarity"""
    try:
        if not query_embedding or not index or not chunks:
            logger.error("Invalid parameters for similarity search")
            return []
            
        if index["size"] == 0:
            logger.error("Empty index for similarity search")
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        embeddings = index["embeddings"]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_array, embeddings)[0]
        
        # Get top k most similar indices with minimum similarity threshold
        similarity_threshold = 0.1
        valid_indices = [(i, sim) for i, sim in enumerate(similarities) if sim > similarity_threshold]
        
        if not valid_indices:
            logger.warning("No chunks meet similarity threshold")
            return chunks[:k]  # Return first k chunks as fallback
        
        # Sort by similarity and take top k
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = valid_indices[:k]
        
        # Return corresponding text chunks
        similar_chunks = []
        for idx, _ in top_indices:
            if 0 <= idx < len(chunks):
                similar_chunks.append(chunks[idx])
        
        logger.info(f"Found {len(similar_chunks)} similar chunks")
        return similar_chunks
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        # Fallback: return first k chunks
        return chunks[:k] if chunks else []
