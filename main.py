# main.py - Updated with lifespan events
import os
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Import after environment is loaded
from config import Config
from server.Routes.hackrx_webhook import router as hackrx_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# In-memory storage for uploaded documents
uploaded_documents = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting HackRx Document Processor API")
    
    # Validate environment variables
    required_vars = ["GEMINI_API_KEY", "HACKRX_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
    else:
        logger.info("✅ All required environment variables configured")
    
    # Log configuration
    logger.info(f"✅ Server starting on port {os.getenv('PORT', 8000)}")
    logger.info(f"✅ Gemini AI: {'Configured' if os.getenv('GEMINI_API_KEY') else 'Not configured'}")
    logger.info(f"✅ Authentication: {'Configured' if os.getenv('HACKRX_TOKEN') else 'Not configured'}")
    
    # Log all registered routes
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            logger.info(f"📍 Route: {route.methods} {route.path}")
    
    yield  # This is where the application runs
    
    # Shutdown
    logger.info("🔄 Shutting down HackRx Document Processor API")
    # Clear uploaded documents if needed
    uploaded_documents.clear()
    logger.info("✅ Cleanup completed")

# Verify environment variables are loaded
print(f"GEMINI_API_KEY loaded: {'✅ Yes' if os.getenv('GEMINI_API_KEY') else '❌ No'}")
print(f"GEMINI_API_KEY length: {len(os.getenv('GEMINI_API_KEY', ''))}")
print(f"HACKRX_TOKEN loaded: {'✅ Yes' if os.getenv('HACKRX_TOKEN') else '❌ No'}")
print(f"PORT: {os.getenv('PORT', 8000)}")

# FastAPI app with lifespan events
app = FastAPI(
    title="HackRx Document Processor API",
    description="Advanced RAG system with Gemini AI for document processing",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan  # ✅ Use lifespan instead of on_event
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
        logger.info("✅ Static files mounted at /static")
    else:
        logger.info("⚠️ Static directory not found - skipping static file mounting")
except Exception as e:
    logger.warning(f"⚠️ Could not mount static files: {e}")

# Security
security = HTTPBearer()
HACKRX_TOKEN = os.getenv("HACKRX_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token for protected endpoints"""
    if not HACKRX_TOKEN:
        logger.error("HACKRX_TOKEN not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured"
        )
    
    if credentials.credentials != HACKRX_TOKEN:
        logger.warning(f"Invalid token attempt: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Pydantic models
class UploadResponse(BaseModel):
    message: str
    filename: str
    file_type: str
    text_preview: str
    file_id: str

class QueryUploadedRequest(BaseModel):
    file_id: str
    questions: List[str]

class APIResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    environment: str
    features: Dict[str, Any]

# Helper functions (your existing functions remain the same)
def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF content with multiple fallback methods"""
    try:
        # Try PyMuPDF first
        try:
            import fitz
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in pdf_doc:
                text += page.get_text() + "\n"
            pdf_doc.close()
            return text.strip()
        except ImportError:
            logger.info("PyMuPDF not available, using PyPDF2")
            
        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except ImportError:
            logger.error("No PDF libraries available")
            raise ValueError("PDF processing libraries not installed")
            
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise ValueError(f"Could not extract text from PDF: {str(e)}")

def extract_docx_text(content: bytes) -> str:
    """Extract text from DOCX content"""
    try:
        from docx import Document
        import io
        doc = Document(io.BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text.strip()
    except ImportError:
        raise ValueError("python-docx library not installed")
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        raise ValueError(f"Could not extract text from DOCX: {str(e)}")

def process_uploaded_file(file_content: bytes, filename: str) -> str:
    """Process uploaded file and extract text"""
    if not file_content:
        raise ValueError("Empty file content")
    
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_content)
    elif file_extension in ['docx']:
        return extract_docx_text(file_content)
    elif file_extension == 'txt':
        try:
            text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = file_content.decode('latin-1')
            except UnicodeDecodeError:
                text = file_content.decode('utf-8', errors='ignore')
        return text.strip()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

# API Router
api_v1_router = APIRouter(prefix="/api/v1", tags=["Document Processing"])

# Your existing endpoints remain the same...
@api_v1_router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process document file"""
    try:
        # Your existing upload logic here
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        allowed_extensions = ['pdf', 'docx', 'txt']
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type '{file_extension}'. Allowed: {', '.join(allowed_extensions)}"
            )
        
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        file_content = await file.read()
        
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        try:
            document_text = process_uploaded_file(file_content, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text content found in the document")
        
        file_id = f"doc_{int(time.time())}_{hash(file.filename) % 10000}"
        
        uploaded_documents[file_id] = {
            "filename": file.filename,
            "content": document_text,
            "upload_time": time.time(),
            "file_type": file_extension,
            "text_length": len(document_text)
        }
        
        logger.info(f"Processed uploaded file: {file.filename} with ID: {file_id} ({len(document_text)} chars)")
        
        return UploadResponse(
            message="File uploaded and processed successfully",
            filename=file.filename,
            file_type=file_extension,
            text_preview=document_text[:500] + "..." if len(document_text) > 500 else document_text,
            file_id=file_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

@api_v1_router.get("/files")
async def list_files():
    """List all uploaded files"""
    try:
        files = []
        for file_id, doc_data in uploaded_documents.items():
            files.append({
                "file_id": file_id,
                "filename": doc_data["filename"],
                "file_type": doc_data["file_type"],
                "upload_time": doc_data["upload_time"],
                "text_length": doc_data.get("text_length", len(doc_data["content"])),
                "status": "processed"
            })
        
        return {
            "uploaded_files": files,
            "total_files": len(files)
        }
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving file list")

@api_v1_router.post("/query-uploaded", response_model=APIResponse)
async def query_uploaded_file(file_id: str = Form(...), questions: str = Form(...)):
    """Query against uploaded file with optimized RAG processing"""
    try:
        question_list = []
        questions = questions.strip()
        
        if not questions:
            raise HTTPException(status_code=400, detail="Questions parameter cannot be empty")
        
        try:
            parsed = json.loads(questions)
            if isinstance(parsed, list):
                question_list = [str(q).strip() for q in parsed if str(q).strip()]
            else:
                raise ValueError("Not a list")
        except (json.JSONDecodeError, ValueError):
            question_list = [q.strip() for q in questions.split(',') if q.strip()]
        
        if not question_list:
            raise HTTPException(status_code=400, detail="No valid questions found in the request")
        
        if len(question_list) > 15:
            raise HTTPException(status_code=400, detail="Too many questions. Maximum 15 questions allowed.")
        
        if file_id not in uploaded_documents:
            logger.error(f"File not found: {file_id}. Available files: {list(uploaded_documents.keys())}")
            raise HTTPException(status_code=404, detail=f"File with ID '{file_id}' not found")
        
        doc_data = uploaded_documents.get(file_id, {})
        document_text = doc_data.get("content", "")
        
        if not document_text:
            raise HTTPException(status_code=400, detail="Document content is empty or unavailable")
        
        try:
            from server.Routes.hackrx_webhook import process_questions_with_advanced_rag
            answers = await process_questions_with_advanced_rag(question_list, document_text)
        except ImportError as e:
            logger.error(f"Import error for RAG processing: {e}")
            answers = [f"Processing error: {str(e)}" for _ in question_list]
        except Exception as e:
            logger.error(f"RAG processing error: {e}")
            answers = [f"Error processing question: {str(e)}" for _ in question_list]
        
        logger.info(f"Processed {len(question_list)} questions for file {file_id}")
        return APIResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error for file_id='{file_id}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error occurred during query processing")

@api_v1_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Railway monitoring"""
    try:
        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            version="3.0.0",
            environment="railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local",
            features={
                "gemini_available": bool(os.getenv("GEMINI_API_KEY")),
                "hackrx_token_available": bool(os.getenv("HACKRX_TOKEN")),
                "file_upload": True,
                "advanced_rag": True,
                "uploaded_files": len(uploaded_documents)
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="degraded",
            timestamp=time.time(),
            version="3.0.0",
            environment="unknown",
            features={"error": str(e)}
        )

# Include routers
app.include_router(hackrx_router, prefix="/api/v1", tags=["HackRx Webhook"])
app.include_router(api_v1_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - Railway health check and API info"""
    try:
        return {
            "status": "online",
            "service": "HackRx Document Processor API",
            "version": "3.0.0",
            "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local",
            "endpoints": {
                "upload_file": "POST /api/v1/upload",
                "list_files": "GET /api/v1/files",
                "query_uploaded": "POST /api/v1/query-uploaded",
                "hackrx_webhook": "POST /api/v1/hackrx/run",
                "health_check": "GET /api/v1/health",
                "api_docs": "GET /docs"
            },
            "webhook_url": {
                "endpoint": "/api/v1/hackrx/run",
                "method": "POST",
                "auth": "Bearer token required"
            }
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return {"status": "error", "message": str(e)}

# Railway app entry point
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"""
🚀 HackRx Document Processor v3.0.0 - Railway Deployment
=======================================================
✅ File Upload Support (/api/v1/upload)
✅ Document Listing (/api/v1/files)
✅ Query Uploaded Files (/api/v1/query-uploaded)  
✅ HackRx Webhook (/api/v1/hackrx/run)
✅ Health Monitoring (/api/v1/health)
✅ API Documentation (/docs)

Server Configuration:
- Host: {host}
- Port: {port}
- Environment: {'Railway' if os.getenv('RAILWAY_ENVIRONMENT') else 'Local'}
- Health Check: /api/v1/health

Environment Status:
- GEMINI_API_KEY: {'✅ Configured' if os.getenv('GEMINI_API_KEY') else '❌ Not configured'}
- HACKRX_TOKEN: {'✅ Configured' if os.getenv('HACKRX_TOKEN') else '❌ Not configured'}

Starting server...
    """)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True,
        loop="asyncio",
        workers=1
    )
