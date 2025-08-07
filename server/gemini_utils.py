# server/gemini_utils.py - High-Accuracy Version
import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {'✅ Yes' if api_key else '❌ No'}")

try:
    import google.generativeai as genai
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    GENAI_AVAILABLE = True
    print("✅ Google GenAI configured successfully")
    
except Exception as e:
    print(f"❌ Error initializing Gemini: {e}")
    GENAI_AVAILABLE = False

def get_text_embedding(text: str) -> List[float]:
    """Get embedding for text using Google Generative AI"""
    if not GENAI_AVAILABLE:
        return []
    
    try:
        # Optimize text length for speed
        text = text[:600] if len(text) > 600 else text
        
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
        
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return []

def get_optimized_answer(question: str, chunks: List[str]) -> str:
    """Get high-accuracy answer using Gemini with optimized prompts"""
    if not GENAI_AVAILABLE:
        # Use pattern matching fallback
        from server.Routes.hackrx_webhook import extract_insurance_answer
        return extract_insurance_answer(question, "\n\n".join(chunks))
    
    try:
        # Ensure chunks are strings
        safe_chunks = []
        for chunk in chunks[:2]:  # Use only top 2 chunks for speed
            if isinstance(chunk, str):
                safe_chunks.append(chunk.strip())
            else:
                safe_chunks.append(str(chunk))
        
        context = "\n\n".join(safe_chunks)
        
        # High-accuracy prompt for insurance questions
        prompt = f"""You are an expert insurance policy analyst. Based ONLY on the provided policy text, answer the question with precise details.

POLICY TEXT:
{context[:1000]}

QUESTION: {question}

REQUIREMENTS:
- Provide exact numbers, percentages, timeframes when mentioned
- Be specific about conditions and requirements
- If information is not in the text, say "Not specified in provided text"
- Keep answer under 200 characters
- Focus on factual policy details

ANSWER:"""
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_k=10,
                top_p=0.9,
                max_output_tokens=100
            )
        )
        
        if response and response.text:
            answer = response.text.strip()
            
            # Clean formatting
            answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
            answer = re.sub(r'\*(.*?)\*', r'\1', answer)
            answer = re.sub(r'\n+', ' ', answer)
            answer = re.sub(r'\s+', ' ', answer)
            
            # Limit length
            if len(answer) > 250:
                answer = answer[:247] + "..."
            
            return answer
        else:
            # Fallback to pattern matching
            from server.Routes.hackrx_webhook import extract_insurance_answer
            return extract_insurance_answer(question, context)
        
    except Exception as e:
        logger.error(f"Error in Gemini answer generation: {e}")
        # Fallback to pattern matching
        from server.Routes.hackrx_webhook import extract_insurance_answer
        return extract_insurance_answer(question, "\n\n".join(chunks))

def filter_insurance_chunks(chunks: List[str], questions: List[str]) -> List[str]:
    """Filter chunks based on insurance relevance and question keywords"""
    # Comprehensive insurance keywords
    insurance_keywords = {
        'premium', 'coverage', 'policy', 'insured', 'claim', 'benefit',
        'waiting', 'period', 'grace', 'deductible', 'copay', 'exclusion',
        'maternity', 'cataract', 'pre-existing', 'ped', 'donor', 'room',
        'rent', 'icu', 'hospital', 'treatment', 'surgery', 'disease',
        'sum', 'amount', 'limit', 'year', 'month', 'day', 'continuous',
        'inception', 'renewal', 'termination', 'conditions', 'terms'
    }
    
    # Extract question-specific keywords
    question_keywords = set()
    for question in questions:
        words = [w.lower().strip('?.,') for w in question.split() if len(w) > 2]
        question_keywords.update(words)
    
    # Combine all relevant keywords
    all_keywords = insurance_keywords.union(question_keywords)
    
    # Score and filter chunks
    scored_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) < 30:  # Skip very short chunks
            continue
            
        chunk_lower = chunk.lower()
        score = 0
        
        # Keyword matching
        for keyword in all_keywords:
            if keyword in chunk_lower:
                score += 2
        
        # Bonus for important phrases
        important_phrases = [
            'grace period', 'waiting period', 'sum insured', 'pre-existing',
            'maternity coverage', 'room rent', 'organ donor', 'cataract surgery',
            'policy conditions', 'continuous coverage', 'policy inception'
        ]
        
        for phrase in important_phrases:
            if phrase in chunk_lower:
                score += 5
        
        # Bonus for numbers (often important in insurance)
        if re.search(r'\d+', chunk):
            score += 3
        
        # Bonus for percentages and currency
        if re.search(r'\d+%|₹|rupees?', chunk, re.IGNORECASE):
            score += 2
        
        # Length consideration
        if 50 <= len(chunk) <= 400:  # Optimal chunk length
            score += 1
        
        if score >= 3:  # Minimum relevance threshold
            scored_chunks.append((chunk, score))
    
    # Sort by relevance and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 25 most relevant chunks
    return [chunk for chunk, score in scored_chunks[:25]]

def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings with error handling"""
    if not GENAI_AVAILABLE:
        logger.warning("Gemini not available for embeddings")
        return [[] for _ in texts]
    
    embeddings = []
    successful = 0
    
    for i, text in enumerate(texts):
        try:
            embedding = get_text_embedding(text)
            if embedding:
                embeddings.append(embedding)
                successful += 1
            else:
                embeddings.append([])
        except Exception as e:
            logger.error(f"Error creating embedding for text {i}: {e}")
            embeddings.append([])
    
    logger.info(f"Created {successful}/{len(texts)} embeddings successfully")
    return embeddings
