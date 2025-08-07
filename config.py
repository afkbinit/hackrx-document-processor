# config.py - Optimized for high accuracy
class Config:
    # Chunk settings optimized for insurance content
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 150
    MAX_TOTAL_CHUNKS = 25
    TOP_K_RESULTS = 3
    
    # Processing settings
    MAX_CONTEXT_LENGTH = 1000
    MAX_ANSWER_LENGTH = 250
    SIMILARITY_THRESHOLD = 0.1
    
    # API settings
    MAX_QUESTIONS = 15
    BATCH_SIZE = 5
    
    # Accuracy settings
    MIN_CHUNK_LENGTH = 30
    MAX_CHUNK_LENGTH = 400
    RELEVANCE_THRESHOLD = 3
