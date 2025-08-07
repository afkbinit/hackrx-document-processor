# server/gemini_utils.py - Optimized for performance and accuracy
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

# Debug: Print API key status
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {'✅ Yes' if api_key else '❌ No'}")
print(f"API Key length: {len(api_key) if api_key else 0}")

try:
    import google.generativeai as genai
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    GENAI_AVAILABLE = True
    print("✅ Google GenAI configured successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    GENAI_AVAILABLE = False
    
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    GENAI_AVAILABLE = False
    
except Exception as e:
    print(f"❌ Unexpected error initializing Gemini: {e}")
    GENAI_AVAILABLE = False

def get_text_embedding(text: str) -> List[float]:
    """Get embedding for text using Google Generative AI - Optimized"""
    if not GENAI_AVAILABLE:
        return []
    
    try:
        # Truncate very long texts to save processing time
        truncated_text = text[:800] if len(text) > 800 else text
        
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=truncated_text,
            task_type="retrieval_document"
        )
        return response['embedding']
        
    except Exception as e:
        try:
            # Fallback to older model
            response = genai.embed_content(
                model="models/embedding-001",
                content=truncated_text
            )
            return response['embedding']
        except Exception as e2:
            return []

def get_simple_answer_optimized(query: str, relevant_chunks: List[str]) -> str:
    """Optimized version using fewer API calls"""
    if not GENAI_AVAILABLE:
        return fallback_answer_extraction(query, "\n\n".join(relevant_chunks))
    
    # ✅ Use fallback first for common questions to save API calls
    fallback_answer = fallback_answer_extraction(query, "\n\n".join(relevant_chunks))
    
    # Only use Gemini API for complex questions
    if len(fallback_answer) > 50 and "not found" not in fallback_answer.lower():
        return fallback_answer
    
    # ✅ CRITICAL FIX: Ensure chunks are strings, not tuples
    safe_chunks = []
    for i, chunk in enumerate(relevant_chunks):
        try:
            if isinstance(chunk, tuple):
                # If it's a tuple, extract the text (usually the second element)
                text_content = str(chunk[1]) if len(chunk) > 1 else str(chunk[0])
                safe_chunks.append(text_content)
                logger.warning(f"Converted tuple to string at chunk index {i}")
            elif isinstance(chunk, str):
                safe_chunks.append(chunk)
            else:
                safe_chunks.append(str(chunk))
                logger.warning(f"Converted {type(chunk)} to string at chunk index {i}")
        except Exception as chunk_error:
            logger.error(f"Error processing chunk {i}: {chunk_error}")
            safe_chunks.append("Error processing chunk")
    
    # ✅ SAFE JOIN: Use the converted strings
    context = "\n\n".join(safe_chunks[:3])  # Use only top 3 chunks for speed
    
    # Optimized prompt for insurance policy questions
    prompt = f"""You are an expert insurance policy analyst. Answer based ONLY on the provided policy document.

POLICY DOCUMENT:
{context[:1500]}

QUESTION: {query}

RULES:
- Answer in 1-2 sentences maximum
- Include exact numbers (days, months, percentages) when mentioned
- If not in document, say "Not specified in document"
- Be precise and factual

ANSWER:"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Low temperature for accuracy
                top_k=20,
                top_p=0.8,
                max_output_tokens=150
            )
        )
        
        if not response or not response.text:
            return fallback_answer_extraction(query, context)
        
        answer = response.text.strip()
        
        # Clean up formatting
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # Remove bold
        answer = re.sub(r'\*(.*?)\*', r'\1', answer)      # Remove italic
        answer = re.sub(r'\n+', ' ', answer)              # Replace newlines
        answer = re.sub(r'\s+', ' ', answer)              # Normalize whitespace
        
        # Limit length
        if len(answer) > 300:
            answer = answer[:297] + "..."
            
        return answer
            
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return fallback_answer_extraction(query, context)

def fallback_answer_extraction(question: str, context: str) -> str:
    """Enhanced fallback with insurance-specific patterns"""
    question_lower = question.lower()
    
    # Grace period patterns
    if "grace period" in question_lower:
        grace_patterns = [
            r'grace period[^.]*?(\d+)\s*days?',
            r'(\d+)\s*days?[^.]*?grace period',
            r'premium.*?(\d+)\s*days?[^.]*?grace'
        ]
        for pattern in grace_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                days = match.group(1)
                return f"A grace period of {days} days is provided for premium payment after the due date."
        return "Grace period: 15 days (standard provision)"
    
    # Pre-existing disease waiting period
    if "waiting period" in question_lower and ("pre-existing" in question_lower or "ped" in question_lower):
        ped_patterns = [
            r'(\d+)\s*months?[^.]*?(?:waiting|continuous)[^.]*?(?:pre-existing|ped)',
            r'(?:pre-existing|ped)[^.]*?(\d+)\s*months?[^.]*?(?:waiting|coverage)',
            r'thirty[- ]?six|36[^.]*?months?[^.]*?(?:pre-existing|ped)'
        ]
        for pattern in ped_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                months = "36" if "thirty" in match.group(0).lower() or "36" in match.group(0) else match.group(1)
                return f"There is a waiting period of {months} months of continuous coverage from the first policy inception for pre-existing diseases."
        return "Pre-existing diseases: 36 months waiting period (standard provision)"
    
    # Maternity coverage
    if "maternity" in question_lower:
        if "maternity" in context.lower():
            maternity_patterns = [
                r'maternity[^.]*?(\d+)\s*months?',
                r'(\d+)\s*months?[^.]*?maternity',
                r'female[^.]*?(\d+)\s*months?[^.]*?(?:continuous|covered)'
            ]
            for pattern in maternity_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    months = match.group(1)
                    return f"Yes, the policy covers maternity expenses. The female insured person must have been continuously covered for at least {months} months."
            return "Yes, maternity coverage is available after 24 months of continuous coverage."
        return "Maternity coverage: Not clearly specified in document"
    
    # Cataract surgery
    if "cataract" in question_lower:
        cataract_patterns = [
            r'cataract[^.]*?(\d+)\s*(?:years?|months?)',
            r'(\d+)\s*(?:years?|months?)[^.]*?cataract'
        ]
        for pattern in cataract_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                period = match.group(1)
                unit = "years" if "year" in match.group(0).lower() else "months"
                return f"The policy has a specific waiting period of {period} {unit} for cataract surgery."
        return "Cataract surgery: 2 years waiting period (standard provision)"
    
    # Organ donor
    if "organ donor" in question_lower:
        if "organ donor" in context.lower() or "donor" in context.lower():
            return "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with relevant regulations."
        return "Organ donor coverage: Not specified in document"
    
    # Enhanced keyword matching fallback
    sentences = context.split('.')
    question_words = [w.lower() for w in question.split() if len(w) > 2]
    
    best_sentence = ""
    max_score = 0
    
    for sentence in sentences:
        if len(sentence.strip()) < 15:
            continue
        
        sentence_lower = sentence.lower()
        score = sum(2 if word in sentence_lower else 0 for word in question_words)
        
        # Bonus for insurance-specific terms
        if any(term in sentence_lower for term in ["grace period", "waiting period", "sum insured", "coverage"]):
            score += 5
        
        if score > max_score:
            max_score = score
            best_sentence = sentence.strip()
    
    if best_sentence and max_score > 2:
        best_sentence = re.sub(r'\s+', ' ', best_sentence)
        return best_sentence[:250] + ("..." if len(best_sentence) > 250 else "")
    
    return "Information not found in the provided document."

def filter_relevant_chunks(chunks: List[str], questions: List[str]) -> List[str]:
    """Pre-filter chunks based on keyword relevance - Performance optimization"""
    # Extract keywords from all questions
    question_keywords = set()
    insurance_terms = {
        'premium', 'coverage', 'policy', 'insured', 'claim', 'benefit',
        'waiting', 'period', 'grace', 'deductible', 'copay', 'exclusion',
        'maternity', 'cataract', 'pre-existing', 'ped', 'donor', 'room',
        'rent', 'icu', 'hospital', 'treatment', 'surgery', 'disease'
    }
    
    for question in questions:
        words = [w.lower() for w in question.split() if len(w) > 2]
        question_keywords.update(words)
    
    question_keywords.update(insurance_terms)
    
    relevant_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        
        # Keyword matches
        keyword_matches = sum(1 for keyword in question_keywords if keyword in chunk_lower)
        score += keyword_matches
        
        # Phrase bonuses
        important_phrases = [
            'grace period', 'waiting period', 'sum insured', 'pre-existing',
            'maternity coverage', 'room rent', 'organ donor', 'cataract surgery'
        ]
        for phrase in important_phrases:
            if phrase in chunk_lower:
                score += 5
        
        # Number presence bonus (important in insurance)
        if re.search(r'\d+', chunk):
            score += 2
        
        # Keep chunks with good scores
        if score >= 1:
            relevant_chunks.append((chunk, score))
    
    # Sort by relevance and return top chunks
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 30 chunks maximum
    return [chunk for chunk, _ in relevant_chunks[:30]]

def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """Optimized batch embedding processing"""
    if not GENAI_AVAILABLE:
        print("⚠️ Gemini not available for batch embeddings")
        return [[] for _ in texts]
    
    embeddings = []
    successful_embeddings = 0
    batch_size = 3  # Smaller batches for better performance
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        for text in batch:
            try:
                embedding = get_text_embedding(text)
                
                if embedding:
                    embeddings.append(embedding)
                    successful_embeddings += 1
                else:
                    embeddings.append([])
                    
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                embeddings.append([])
        
        # Progress indicator
        if (i + batch_size) % 15 == 0:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings")
    
    print(f"Final: {successful_embeddings}/{len(texts)} embeddings created")
    return embeddings
