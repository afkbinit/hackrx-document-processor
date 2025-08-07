# server/Routes/hackrx_webhook.py - Original working version
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

# Original processing function
async def process_questions_with_advanced_rag(questions: List[str], document_text: str) -> List[str]:
    """Original RAG processing"""
    try:
        logger.info(f"Processing {len(questions)} questions with RAG")
        
        # Step 1: Create chunks
        all_chunks = smart_chunk_text(
            document_text, 
            chunk_size=Config.CHUNK_SIZE, 
            overlap=Config.CHUNK_OVERLAP
        )
        
        if not all_chunks:
            logger.error("No chunks created from document")
            return ["No content found in the document"] * len(questions)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Step 2: Filter chunks
        relevant_chunks = filter_relevant_chunks(all_chunks, questions)
        if not relevant_chunks:
            relevant_chunks = all_chunks[:Config.MAX_TOTAL_CHUNKS]
        
        logger.info(f"Filtered to {len(relevant_chunks)} relevant chunks")
        
        # Step 3: Create embeddings
        chunk_embeddings = batch_get_embeddings(relevant_chunks)
        
        if not chunk_embeddings or not any(chunk_embeddings):
            logger.error("Failed to create embeddings")
            return await process_questions_simple_fallback(questions, document_text)
        
        # Step 4: Create FAISS index
        faiss_index = create_faiss_index(chunk_embeddings)
        if faiss_index is None:
            logger.error("Failed to create FAISS index")
            return await process_questions_simple_fallback(questions, document_text)
        
        # Step 5: Process questions
        answers = []
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}")
                
                # Get question embedding
                question_embedding = batch_get_embeddings([question])[0]
                if not question_embedding:
                    answers.append("Failed to process question")
                    continue
                
                # Find similar chunks - ORIGINAL WORKING VERSION
                similar_chunks = search_similar_chunks(
                    query_embedding=question_embedding,
                    index=faiss_index,
                    chunks=relevant_chunks,
                    k=Config.TOP_K_RESULTS
                )
                
                if not similar_chunks:
                    answers.append("No relevant information found")
                    continue
                
                # Generate answer
                answer = get_simple_answer(question, similar_chunks)
                answers.append(answer)
                
                logger.info(f"Generated answer for question {i+1}")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        return answers
        
    except Exception as e:
        logger.error(f"Error in RAG processing: {str(e)}")
        return await process_questions_simple_fallback(questions, document_text)

async def process_questions_simple_fallback(questions: List[str], document_text: str) -> List[str]:
    """Fallback processing"""
    from server.gemini_utils import fallback_answer_extraction
    
    answers = []
    for question in questions:
        try:
            answer = fallback_answer_extraction(question, document_text)
            answers.append(answer)
        except Exception as e:
            answers.append("Error processing this question.")
    
    return answers

@router.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, token: str = Depends(verify_token)):
    """Original HackRx endpoint"""
    try:
        start_time = time.time()
        
        if not request.documents or not request.questions:
            raise HTTPException(status_code=400, detail="Documents and questions required")
        
        # Extract text from PDF
        document_text = extract_text_from_pdf_url(request.documents)
        if not document_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from document")
        
        # Process questions
        answers = await process_questions_with_advanced_rag(request.questions, document_text)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return HackRxResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in hackrx_run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
