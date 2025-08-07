import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HACKRX_TOKEN = os.getenv("HACKRX_TOKEN")
    PORT = int(os.getenv("PORT", 8000))
    
    # Optional - Pinecone (for general embedding service)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")
    
    # HackRx Configuration
    HACKRX_API_URL = "http://localhost:8000/api/v1"
    
    # Model Configuration - Optimized for performance
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # For Pinecone
    GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"       # For Gemini
    LLM_MODEL = "gemini-1.5-flash"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.1
    
    # Performance Settings - Optimized
    CHUNK_SIZE = 1500              # Larger chunks = fewer total chunks
    CHUNK_OVERLAP = 100            # Reduced overlap
    TOP_K_RESULTS = 3              # Fewer results per search
    MAX_CHUNKS_PER_QUESTION = 20   # Limit chunks processed per question
    MAX_TOTAL_CHUNKS = 60          # Maximum chunks for any document
    
    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        print(f"✅ GEMINI_API_KEY loaded: {len(cls.GEMINI_API_KEY)} characters")
        
        if not cls.HACKRX_TOKEN:
            raise ValueError("HACKRX_TOKEN not found in environment variables")
        print(f"✅ HACKRX_TOKEN loaded: Available")
        
        if cls.PINECONE_API_KEY:
            print(f"✅ PINECONE_API_KEY loaded: Available")
        else:
            print(f"⚠️ PINECONE_API_KEY: Not configured (fallback mode)")

# Create settings instance for backward compatibility
settings = Config()
