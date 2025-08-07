from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ClauseReference(BaseModel):
    clause_id: str
    content: str
    relevance_score: float
    page_number: Optional[int] = None

class DecisionResponse(BaseModel):
    decision: str
    amount: Optional[float] = None
    confidence: float
    justification: str
    referenced_clauses: List[ClauseReference]

class QueryResponse(BaseModel):
    query: str
    answer: str
    decision: DecisionResponse
    processing_time: float
    token_usage: int

class APIResponse(BaseModel):
    answers: List[str]
