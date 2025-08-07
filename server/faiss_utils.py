# server/faiss_utils.py
import faiss
import numpy as np
from typing import List

def create_faiss_index(embeddings: List[List[float]]):
    """
    Creates a FAISS index from a list of embeddings (768-d float vectors).
    Returns the index.
    """
    if not embeddings:
        return None
        
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)  # L2 = Euclidean distance

    # Convert to NumPy array of shape (n_chunks, 768)
    vectors_np = np.array(embeddings).astype("float32")
    index.add(vectors_np)

    return index

def search_similar_chunks(query_embedding: List[float], index, chunks: List[str], k: int = 5) -> List[str]:
    """
    Returns top-k most relevant chunks for a given query embedding.
    """
    if not query_embedding or not index or not chunks:
        return []
        
    query_np = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_np, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):  # Safety check
            results.append(chunks[idx])

    return results
