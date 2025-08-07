# config.py - Optimized configuration
class Config:
    # Optimized chunk settings for accuracy and speed
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    MAX_TOTAL_CHUNKS = 50
    TOP_K_RESULTS = 5
    
    # Performance settings
    MAX_CONTEXT_LENGTH = 1500
    MAX_ANSWER_LENGTH = 300
    SIMILARITY_THRESHOLD = 0.3
    
    # Processing limits
    MAX_QUESTIONS = 15
    BATCH_SIZE = 3
