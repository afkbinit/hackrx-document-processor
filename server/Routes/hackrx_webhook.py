# server/Routes/hackrx_webhook.py - Fixed version with all optimizations
import os
import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from server.pdf_utils import extract_text_from_pdf_url
from server.chunker import smart_chunk_text
from server.faiss_utils import create_faiss_index, search_similar_chunks
from server.gemini_utils import get_simple_answer, batch_get_embeddings, filter_relevant_chunks
from config import Config

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()
HACKRX_TOKEN = os.getenv("HACKRX_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Models
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# Optimized processing function
async def process_questions_with_advanced_rag(questions: List[str], document_text: str) -> List[str]:
    """Optimized RAG processing with performance improvements"""
    try:
        logger.info(f"Processing {len(questions)} questions with optimized RAG")
        
        # Step 1: Create optimized chunks
        all_chunks = smart_chunk_text(
            document_text, 
            chunk_size=800,  # Optimized chunk size
            overlap=200      # Increased overlap
        )
        
        if not all_chunks:
            logger.error("No chunks created from document")
            return ["No content found in the document"] * len(questions)
        
        logger.info(f"Created {len(all_chunks)} initial chunks")
        
        # Step 2: Pre-filter chunks for relevance
        relevant_chunks = filter_relevant_chunks(all_chunks, questions)
        
        if not relevant_chunks:
            relevant_chunks = all_chunks[:50]  # Increased limit
        
        logger.info(f"Filtered to {len(relevant_chunks)} relevant chunks from {len(all_chunks)} total")
        
        # Step 3: Create embeddings for filtered chunks only
        logger.info("Creating embeddings for filtered chunks")
        chunk_embeddings = batch_get_embeddings(relevant_chunks)
        
        if not chunk_embeddings or not any(chunk_embeddings):
            logger.error("Failed to create embeddings")
            return await process_questions_simple_fallback(questions, document_text)
        
        logger.info("Creating FAISS index")
        faiss_index = create_faiss_index(chunk_embeddings)
        
        if faiss_index is None:
            logger.error("Failed to create FAISS index")
            return await process_questions_simple_fallback(questions, document_text)
        
        # Step 4: Process each question efficiently
        answers = []
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                # Get question embedding
                question_embeddings = batch_get_embeddings([question])
                if not question_embeddings or not question_embeddings[0]:
                    answers.append("Failed to process question")
                    continue
                
                question_embedding = question_embeddings[0]
                
                # ✅ FIXED: Include all required parameters
                similar_chunks = search_similar_chunks(
                    query_embedding=question_embedding,
                    index=faiss_index,
                    chunks=relevant_chunks,  # ← Fixed: Added missing parameter
                    k=5
                )
                
                if not similar_chunks:
                    answers.append("No relevant information found in the document")
                    continue
                
                # Generate answer using optimized Gemini
                answer = get_simple_answer(question, similar_chunks)
                answers.append(answer)
                
                logger.info(f"Generated answer for question {i+1}")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                answers.append(f"Error processing question: {str(e)}")
        
        return answers
        
    except Exception as e:
        logger.error(f"Error in optimized RAG processing: {str(e)}")
        return await process_questions_simple_fallback(questions, document_text)

async def process_questions_simple_fallback(questions: List[str], document_text: str) -> List[str]:
    """Fallback processing using simple text matching"""
    logger.info("Using simple fallback processing")
    
    from server.gemini_utils import fallback_answer_extraction
    
    answers = []
    for question in questions:
        try:
            answer = fallback_answer_extraction(question, document_text)
            answers.append(answer)
        except Exception as e:
            logger.error(f"Error in fallback processing: {e}")
            answers.append("Error processing this question.")
    
    return answers

# Main processing pipeline
async def process_document_questions(document_url: str, questions: List[str]) -> List[str]:
    """Main processing pipeline for document URL"""
    try:
        # Extract text from PDF
        logger.info(f"Extracting text from PDF: {document_url}")
        document_text = extract_text_from_pdf_url(document_url)
        
        if not document_text:
            logger.error("Failed to extract text from PDF")
            return ["Failed to extract text from the document"] * len(questions)
        
        logger.info(f"Extracted {len(document_text)} characters from PDF")
        
        # Use optimized processing
        return await process_questions_with_advanced_rag(questions, document_text)
        
    except Exception as e:
        logger.error(f"Error in main processing pipeline: {str(e)}")
        return [f"Error processing document: {str(e)}"] * len(questions)

# Main HackRx endpoint
@router.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """Optimized HackRx endpoint for fast, accurate processing"""
    try:
        start_time = time.time()
        
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        if len(request.questions) > 15:
            raise HTTPException(status_code=400, detail="Too many questions (maximum 15)")
        
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        # Process the request with optimizations
        answers = await process_document_questions(request.documents, request.questions)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return HackRxResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hackrx_run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check
@router.get("/router-health")
async def router_health():
    """Health check for HackRx router"""
    return {
        "status": "healthy",
        "router": "hackrx_webhook",
        "gemini_available": os.getenv("GEMINI_API_KEY") is not None,
        "timestamp": time.time()
    }
