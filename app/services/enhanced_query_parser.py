
# app/services/enhanced_query_parser.py
import openai
import re
import json
from typing import Dict, Any, Optional, List
from app.models.request_models import ProcessedQuery
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedQueryParser:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        
        # Insurance-specific patterns
        self.age_patterns = [
            r'(\d{1,2})\s*(?:year|yr|y)?(?:old|yo)?',
            r'(\d{1,2})\s*[MF]',
            r'age\s*(\d{1,2})'
        ]
        
        self.gender_patterns = {
            'M': r'\b(?:male|man|M|boy)\b',
            'F': r'\b(?:female|woman|F|girl)\b'
        }
        
        self.procedure_patterns = {
            'knee surgery': r'\b(?:knee|ACL|meniscus)\s*(?:surgery|operation|repair)',
            'cataract': r'\b(?:cataract|eye)\s*(?:surgery|operation)',
            'heart surgery': r'\b(?:heart|cardiac|CABG|bypass)\s*(?:surgery|operation)',
            'dialysis': r'\b(?:dialysis|kidney|renal)\b',
            'maternity': r'\b(?:maternity|pregnancy|delivery|childbirth)\b'
        }
        
        self.location_patterns = [
            r'\b(?:in|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:hospital|clinic)\b'
        ]
        
    async def parse_query(self, query: str) -> ProcessedQuery:
        """Enhanced parsing for insurance queries"""
        try:
            # First try LLM parsing with insurance context
            prompt = self._create_insurance_parsing_prompt(query)
            
            response = await openai.ChatCompletion.acreate(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._get_insurance_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent parsing
                max_tokens=800
            )
            
            parsed_data = json.loads(response.choices[0].message.content)
            
            # Validate and enhance with regex parsing
            enhanced_data = self._enhance_with_regex(query, parsed_data)
            
            return ProcessedQuery(
                original_query=query,
                age=enhanced_data.get('age'),
                gender=enhanced_data.get('gender'),
                procedure=enhanced_data.get('procedure'),
                location=enhanced_data.get('location'),
                policy_duration=enhanced_data.get('policy_duration'),
                entities=enhanced_data.get('entities', {})
            )
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {str(e)}, falling back to regex")
            return self._regex_parse(query)
    
    def _get_insurance_system_prompt(self) -> str:
        return """You are an expert insurance claim processor specializing in health insurance policies. 
        You understand medical procedures, policy terms, waiting periods, and claim eligibility criteria.
        Extract information accurately for insurance claim processing."""
    
    def _create_insurance_parsing_prompt(self, query: str) -> str:
        return f"""
        Parse this insurance/medical query for claim processing:
        Query: "{query}"
        
        Extract and return JSON with these fields:
        {{
            "age": <integer or null>,
            "gender": <"M"/"F" or null>,
            "procedure": <specific medical procedure or null>,
            "location": <city name or null>,
            "policy_duration": <duration in months/years or null>,
            "policy_type": <type if mentioned or null>,
            "urgency": <"emergency"/"planned" or null>,
            "entities": {{
                "medical_conditions": [],
                "time_references": [],
                "financial_amounts": [],
                "policy_details": [],
                "waiting_periods": []
            }}
        }}
        
        Focus on:
        - Medical procedures (surgery, dialysis, maternity, etc.)
        - Age and gender for eligibility
        - Location for network provider checks
        - Policy duration for waiting period calculations
        - Emergency vs planned treatment
        """
    
    def _enhance_with_regex(self, query: str, llm_data: dict) -> dict:
        """Enhance LLM output with regex patterns"""
        enhanced = llm_data.copy()
        
        # Age extraction with multiple patterns
        if not enhanced.get('age'):
            for pattern in self.age_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    try:
                        enhanced['age'] = int(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Gender extraction
        if not enhanced.get('gender'):
            for gender, pattern in self.gender_patterns.items():
                if re.search(pattern, query, re.IGNORECASE):
                    enhanced['gender'] = gender
                    break
        
        # Procedure extraction
        if not enhanced.get('procedure'):
            for procedure, pattern in self.procedure_patterns.items():
                if re.search(pattern, query, re.IGNORECASE):
                    enhanced['procedure'] = procedure
                    break
        
        # Location extraction
        if not enhanced.get('location'):
            for pattern in self.location_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    enhanced['location'] = match.group(1)
                    break
        
        return enhanced
    
    def _regex_parse(self, query: str) -> ProcessedQuery:
        """Fallback regex-only parsing"""
        entities = {
            "medical_conditions": [],
            "time_references": [],
            "financial_amounts": [],
            "policy_details": [],
            "waiting_periods": []
        }
        
        # Extract all entities using regex
        age = None
        for pattern in self.age_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    age = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        gender = None
        for g, pattern in self.gender_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                gender = g
                break
        
        return ProcessedQuery(
            original_query=query,
            age=age,
            gender=gender,
            entities=entities
        )
