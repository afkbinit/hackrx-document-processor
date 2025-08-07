# server/prompt_utils.py
from typing import List

def build_prompt(query: str, relevant_chunks: List[str]) -> str:
    """
    Build a comprehensive prompt for Gemini to answer insurance policy questions
    """
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are an expert insurance policy analyst with deep knowledge of Indian insurance regulations and policy terms. Your task is to provide accurate, specific answers based strictly on the provided policy document context.

**CONTEXT FROM POLICY DOCUMENT:**
{context}

**QUESTION:** {query}

**INSTRUCTIONS:**
1. Answer based ONLY on the information provided in the context above
2. Be specific and include exact details like time periods, percentages, amounts when mentioned
3. If the information is not in the provided context, state "Information not found in the provided document"
4. Use clear, professional language
5. Include relevant conditions, exclusions, or requirements when applicable

**ANSWER:**"""

    return prompt

def build_structured_prompt(query: str, relevant_chunks: List[str]) -> str:
    """
    Build a prompt that encourages structured JSON responses
    """
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are an expert insurance policy analyst. Analyze the provided policy document context and answer the question with structured information.

**POLICY CONTEXT:**
{context}

**QUESTION:** {query}

Please provide your answer in the following JSON format:
{{
    "answer": "Your detailed answer here",
    "confidence": "high/medium/low",
    "key_details": ["detail1", "detail2", "detail3"],
    "conditions": ["condition1", "condition2"] (if any),
    "source_reference": "brief reference to policy section"
}}

**IMPORTANT:**
- Base your answer ONLY on the provided context
- Be specific with numbers, time periods, and conditions
- If information is not available, set confidence to "low" and mention in answer
- Include exact policy terms and conditions when available

Response:"""

    return prompt
