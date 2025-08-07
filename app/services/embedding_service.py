import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        # Initialize Pinecone only if available
        self.pinecone_available = bool(Config.PINECONE_API_KEY)
        
        if self.pinecone_available:
            try:
                pinecone.init(
                    api_key=Config.PINECONE_API_KEY,
                    environment=Config.PINECONE_ENVIRONMENT
                )
                self._initialize_index()
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {e}")
                self.pinecone_available = False
                self.index = None
                self.vectors_store = {}
        else:
            logger.info("Pinecone not configured, using in-memory storage")
            self.index = None
            self.vectors_store = {}
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info(f"✅ SentenceTransformer model loaded: {Config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            self.model = None
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        try:
            if Config.PINECONE_INDEX_NAME not in pinecone.list_indexes():
                pinecone.create_index(
                    Config.PINECONE_INDEX_NAME,
                    dimension=384,  # MiniLM embedding dimension
                    metric='cosine'
                )
            
            self.index = pinecone.Index(Config.PINECONE_INDEX_NAME)
            logger.info("✅ Pinecone index initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            self.index = None
            self.vectors_store = {}
    
    async def store_document_embeddings(self, document_data: Dict[str, Any]) -> bool:
        """Store document chunks as embeddings"""
        if not self.model:
            logger.error("SentenceTransformer model not available")
            return False
            
        try:
            chunks = document_data['chunks']
            vectors = []
            
            # Process chunks in batches for better performance
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                for chunk in batch_chunks:
                    try:
                        embedding = self.model.encode(chunk['text'])
                        
                        vector_data = {
                            'id': f"{document_data['url']}_{chunk['id']}",
                            'values': embedding.tolist(),
                            'metadata': {
                                'text': chunk['text'][:1000],
                                'document_url': document_data['url'],
                                'chunk_id': chunk['id'],
                                'clause_patterns': chunk.get('clause_patterns', []),
                                'word_count': chunk.get('word_count', 0)
                            }
                        }
                        vectors.append(vector_data)
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk.get('id', 'unknown')}: {e}")
                        continue
            
            # Store vectors
            if self.index and vectors:
                # Store in Pinecone
                try:
                    self.index.upsert(vectors)
                    logger.info(f"✅ Stored {len(vectors)} embeddings in Pinecone")
                except Exception as e:
                    logger.error(f"Pinecone upsert failed: {e}")
                    # Fallback to in-memory storage
                    for vector in vectors:
                        self.vectors_store[vector['id']] = vector
            else:
                # Store in memory
                for vector in vectors:
                    self.vectors_store[vector['id']] = vector
                logger.info(f"✅ Stored {len(vectors)} embeddings in memory")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False
    
    async def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar document chunks"""
        if not self.model:
            logger.error("SentenceTransformer model not available")
            return []
            
        try:
            query_embedding = self.model.encode(query)
            
            if self.index:
                # Search in Pinecone
                try:
                    results = self.index.query(
                        vector=query_embedding.tolist(),
                        top_k=top_k,
                        include_metadata=True
                    )
                    
                    return [
                        {
                            'id': match.id,
                            'score': match.score,
                            'text': match.metadata.get('text', ''),
                            'document_url': match.metadata.get('document_url'),
                            'chunk_id': match.metadata.get('chunk_id'),
                            'clause_patterns': match.metadata.get('clause_patterns', [])
                        }
                        for match in results.matches
                    ]
                except Exception as e:
                    logger.error(f"Pinecone query failed: {e}")
                    # Fallback to in-memory search
                    return self._fallback_search(query_embedding, top_k)
            else:
                # In-memory search
                return self._fallback_search(query_embedding, top_k)
                
        except Exception as e:
            logger.error(f"Error searching embeddings: {str(e)}")
            return []
    
    def _fallback_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Fallback similarity search using in-memory storage"""
        if not self.vectors_store:
            return []
            
        similarities = []
        
        for vector_id, vector_data in self.vectors_store.items():
            try:
                stored_embedding = np.array(vector_data['values'])
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                
                similarities.append({
                    'id': vector_id,
                    'score': float(similarity),
                    'metadata': vector_data['metadata']
                })
            except Exception as e:
                logger.error(f"Error computing similarity for {vector_id}: {e}")
                continue
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        return [
            {
                'id': sim['id'],
                'score': sim['score'],
                'text': sim['metadata'].get('text', ''),
                'document_url': sim['metadata'].get('document_url'),
                'chunk_id': sim['metadata'].get('chunk_id'),
                'clause_patterns': sim['metadata'].get('clause_patterns', [])
            }
            for sim in similarities[:top_k]
        ]
