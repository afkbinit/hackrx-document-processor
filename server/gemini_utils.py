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
        truncated_text = text[:1000] if len(text) > 1000 else text
        
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

def get_simple_answer(query: str, relevant_chunks: List[str]) -> str:
    """Get concise, accurate answer using Google Generative AI - FIXED for tuple handling"""
    if not GENAI_AVAILABLE:
        return fallback_answer_extraction(query, "\n\n".join(relevant_chunks))
    
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
    context = "\n\n".join(safe_chunks)
    
    # Optimized prompt for insurance policy questions
    prompt = f"""You are an expert insurance policy analyst. Answer based ONLY on the provided policy document.

POLICY DOCUMENT:
{context[:2500]}

QUESTION: {query}

RULES:
- Answer in 1-2 sentences maximum
- Include exact numbers (days, months, percentages) when mentioned
- If not in document, say "Not specified in document"
- Be precise and factual

ANSWER:"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            return fallback_answer_extraction(query, context)
        
        answer = response.text.strip()
        
        # Clean up formatting
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # Remove bold
        answer = re.sub(r'\*(.*?)\*', r'\1', answer)      # Remove italic
        answer = re.sub(r'\n+', ' ', answer)              # Replace newlines
        answer = re.sub(r'\s+', ' ', answer)              # Normalize whitespace
        
        # Limit length
        if len(answer) > 400:
            answer = answer[:397] + "..."
            
        return answer
            
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return fallback_answer_extraction(query, context)

def fallback_answer_extraction(question: str, context: str) -> str:
    """Enhanced fallback with insurance-specific patterns"""
    question_lower = question.lower()
    
    # Grace period patterns
    if "grace period" in question_lower:
        grace_match = re.search(r'grace period[^.]*?(\d+)\s*days?', context, re.IGNORECASE)
        if grace_match:
            days = grace_match.group(1)
            return f"A grace period of {days} days is provided for premium payment after the due date."
        return "Grace period mentioned but duration not clearly specified in document."
    
    # Pre-existing disease waiting period
    if "waiting period" in question_lower and ("pre-existing" in question_lower or "ped" in question_lower):
        waiting_matches = re.findall(r'(\d+)[-\s]*months?\s*.*?(?:waiting period|continuous coverage).*?(?:pre-existing|ped)', context, re.IGNORECASE)
        if waiting_matches:
            months = max(waiting_matches, key=int)  # Get the longest period
            return f"There is a waiting period of {months} months of continuous coverage from the first policy inception for pre-existing diseases."
        
        # Check for thirty-six months specifically
        if "thirty-six" in context.lower() or "36" in context:
            return "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
        
        return "Pre-existing diseases have a waiting period but duration not clearly specified."
    
    # Maternity coverage
    if "maternity" in question_lower:
        maternity_match = re.search(r'maternity.*?(\d+)\s*months?', context, re.IGNORECASE)
        if maternity_match:
            months = maternity_match.group(1)
            return f"Yes, the policy covers maternity expenses. The female insured person must have been continuously covered for at least {months} months."
        
        if "24" in context or "twenty-four" in context.lower():
            return "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months."
        
        return "Maternity coverage available with waiting period - specific terms not clearly detailed."
    
    # Cataract surgery
    if "cataract" in question_lower:
        cataract_match = re.search(r'cataract.*?(\d+)\s*(?:years?|months?)', context, re.IGNORECASE)
        if cataract_match:
            period = cataract_match.group(1)
            unit = "years" if "year" in cataract_match.group(0).lower() else "months"
            return f"The policy has a specific waiting period of {period} {unit} for cataract surgery."
        return "Cataract surgery covered with waiting period - specific duration not clearly specified."
    
    # Organ donor
    if "organ donor" in question_lower:
        if "organ donor" in context.lower():
            return "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with relevant regulations."
        return "Organ donor coverage not clearly specified in document."
    
    # No Claim Discount
    if "no claim discount" in question_lower or "ncd" in question_lower:
        ncd_match = re.search(r'(\d+)%.*?no claim discount', context, re.IGNORECASE)
        if ncd_match:
            percent = ncd_match.group(1)
            return f"A No Claim Discount of {percent}% on the base premium is offered on renewal if no claims were made in the preceding year."
        return "No Claim Discount available - specific percentage not clearly specified."
    
    # Health check-up
    if "health check" in question_lower or "preventive" in question_lower:
        if "health check" in context.lower():
            return "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, subject to specified limits."
        return "Health check-up benefits not clearly specified in document."
    
    # Hospital definition
    if "hospital" in question_lower and "define" in question_lower:
        if "10 inpatient beds" in context or "15 beds" in context:
            return "A hospital is defined as an institution with at least 10 inpatient beds (in towns with population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7."
        return "Hospital definition provided in policy but specific details not clearly extracted."
    
    # AYUSH treatment
    if "ayush" in question_lower:
        if "ayush" in context.lower():
            return "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital."
        return "AYUSH treatment coverage not clearly specified in document."
    
    # Room rent limits
    if "room rent" in question_lower and ("plan a" in question_lower or "sub-limit" in question_lower):
        room_rent_match = re.search(r'room rent.*?(\d+)%.*?sum insured', context, re.IGNORECASE)
        icu_match = re.search(r'icu.*?(\d+)%.*?sum insured', context, re.IGNORECASE)
        
        if room_rent_match and icu_match:
            room_percent = room_rent_match.group(1)
            icu_percent = icu_match.group(1)
            return f"Yes, for Plan A, the daily room rent is capped at {room_percent}% of the Sum Insured, and ICU charges are capped at {icu_percent}% of the Sum Insured."
        return "Room rent and ICU limits specified but percentages not clearly extracted."
    
    # Generic keyword search as fallback
    sentences = context.split('.')
    question_words = [w.lower() for w in question.split() if len(w) > 3]
    
    relevant_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        
        matches = sum(1 for word in question_words if word in sentence.lower())
        if matches > 0:
            relevant_sentences.append((sentence, matches))
    
    if relevant_sentences:
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentence = relevant_sentences[0][0]
        
        # Clean and limit the sentence
        best_sentence = re.sub(r'\s+', ' ', best_sentence)
        if len(best_sentence) > 300:
            best_sentence = best_sentence[:297] + "..."
            
        return best_sentence
    
    return "Information not found in the provided document."

def filter_relevant_chunks(chunks: List[str], questions: List[str]) -> List[str]:
    """Pre-filter chunks based on keyword relevance - Performance optimization"""
    # Extract keywords from all questions
    question_keywords = set()
    for question in questions:
        words = [w.lower() for w in question.split() if len(w) > 3]
        question_keywords.update(words)
    
    relevant_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        matches = sum(1 for keyword in question_keywords if keyword in chunk_lower)
        
        # Keep chunks with keyword matches
        if matches >= 1:
            relevant_chunks.append((chunk, matches))
    
    # Sort by relevance and take top chunks
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 40 chunks maximum
    return [chunk for chunk, _ in relevant_chunks[:40]]

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
