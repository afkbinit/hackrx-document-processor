from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class ProcessedQuery(BaseModel):
    original_query: str
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    entities: dict = {}
