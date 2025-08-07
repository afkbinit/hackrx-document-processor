# app/services/enhanced_decision_engine.py
import openai
import json
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from app.models.response_models import DecisionResponse, ClauseReference
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedDecisionEngine:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        
        # Insurance business rules
        self.waiting_periods = {
            'general': 30,  # days
            'pre_existing': 48,  # months
            'specific_diseases': 24,  # months
            'maternity': 9,  # months
        }
        
        self.procedure_categories = {
            'emergency': ['accident', 'heart attack', 'stroke', 'emergency'],
            'planned': ['cataract', 'knee surgery', 'hernia'],
            'maternity': ['delivery', 'pregnancy', 'childbirth'],
            'dental': ['tooth', 'dental', 'oral surgery']
        }
    
    async def make_decision(self, query: str, relevant_chunks: List[Dict[str, Any]], 
                          parsed_query: Dict[str, Any]) -> DecisionResponse:
        """Enhanced decision making with insurance business logic"""
        try:
            # Pre-process for insurance-specific logic
            policy_context = self._build_policy_context(relevant_chunks)
            eligibility_check = self._check_eligibility(parsed_query, policy_context)
            
            # Create comprehensive decision prompt
            prompt = self._create_enhanced_decision_prompt(query, policy_context, parsed_query, eligibility_check)
            
            response = await openai.ChatCompletion.acreate(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._get_insurance_expert_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for consistent decisions
                max_tokens=1500
            )
            
            decision_data = json.loads(response.choices[0].message.content)
            
            # Validate and enhance decision
            enhanced_decision = self._enhance_decision(decision_data, eligibility_check, policy_context)
            
            # Create detailed clause references
            clause_refs = self._create_detailed_clause_references(relevant_chunks, enhanced_decision)
            
            return DecisionResponse(
                decision=enhanced_decision.get('decision', 'pending'),
                amount=enhanced_decision.get('amount'),
                confidence=enhanced_decision.get('confidence', 0.8),
                justification=enhanced_decision.get('justification', ''),
                referenced_clauses=clause_refs
            )
            
        except Exception as e:
            logger.error(f"Decision making error: {str(e)}")
            return self._create_safe_fallback_decision(query, relevant_chunks, parsed_query)
    
    def _get_insurance_expert_prompt(self) -> str:
        return """You are a senior insurance claim processor with 20+ years of experience in health insurance. 
        You are an expert in:
        - Policy terms and conditions interpretation
        - Waiting period calculations
        - Exclusion analysis
        - Medical procedure coverage assessment
        - Claim eligibility determination
        
        Always provide accurate, fair, and well-reasoned decisions based on policy terms.
        Be conservative but fair in your assessments. If information is insufficient, state so clearly.
        Reference specific policy clauses in your reasoning."""
    
    def _create_enhanced_decision_prompt(self, query: str, policy_context: Dict, 
                                       parsed_query: Dict, eligibility_check: Dict) -> str:
        return f"""
        INSURANCE CLAIM ASSESSMENT
        
        QUERY: {query}
        
        PARSED INFORMATION:
        - Age: {parsed_query.get('age')}
        - Gender: {parsed_query.get('gender')}
        - Procedure: {parsed_query.get('procedure')}
        - Location: {parsed_query.get('location')}
        - Policy Duration: {parsed_query.get('policy_duration')}
        
        ELIGIBILITY PRE-CHECK:
        {json.dumps(eligibility_check, indent=2)}
        
        RELEVANT POLICY CONTEXT:
        {policy_context.get('key_clauses', 'No specific clauses found')}
        
        WAITING PERIODS IDENTIFIED:
        {policy_context.get('waiting_periods', 'Not specified')}
        
        COVERAGE LIMITS:
        {policy_context.get('coverage_limits', 'Not specified')}
        
        EXCLUSIONS APPLICABLE:
        {policy_context.get('exclusions', 'None identified')}
        
        Based on the above information, provide a comprehensive claim decision in JSON format:
        {{
            "decision": "approved|rejected|partial|pending_documents",
            "decision_code": "specific_reason_code",
            "amount": <claim_amount_if_applicable>,
            "confidence": <0.0_to_1.0>,
            "justification": "<detailed_explanation_with_clause_references>",
            "waiting_period_status": "<satisfied|not_satisfied|not_applicable>",
            "exclusions_applicable": ["list_of_applicable_exclusions"],
            "additional_requirements": ["list_of_additional_documents_needed"],
            "reasoning_steps": [
                "<step_1_assessment>",
                "<step_2_policy_check>",
                "<step_3_eligibility>",
                "<step_4_conclusion>"
            ]
        }}
        
        IMPORTANT GUIDELINES:
        1. If waiting period is not satisfied, decision should be "rejected"
        2. If procedure is excluded, decision should be "rejected"
        3. If information is insufficient, decision should be "pending_documents"
        4. Reference specific clause numbers where possible
        5. Be precise about monetary amounts and percentages
        """
    
    def _build_policy_context(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive policy context from chunks"""
        context = {
            'key_clauses': [],
            'waiting_periods': [],
            'coverage_limits': [],
            'exclusions': [],
            'benefits': []
        }
        
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'general')
            text = chunk['text']
            
            if chunk_type == 'waiting_period':
                periods = re.findall(r'(\d+)\s*(days?|months?|years?)', text, re.IGNORECASE)
                context['waiting_periods'].extend(periods)
            
            elif chunk_type == 'exclusion':
                exclusions = self._extract_exclusions(text)
                context['exclusions'].extend(exclusions)
            
            elif chunk_type == 'benefit':
                benefits = self._extract_benefits(text)
                context['benefits'].extend(benefits)
            
            elif chunk_type == 'financial':
                limits = self._extract_financial_limits(text)
                context['coverage_limits'].extend(limits)
            
            # Add high-importance chunks to key clauses
            if chunk.get('importance_score', 0) > 2.0:
                context['key_clauses'].append({
                    'text': text[:500],
                    'score': chunk['importance_score'],
                    'type': chunk_type
                })
        
        return context
    
    def _check_eligibility(self, parsed_query: Dict, policy_context: Dict) -> Dict[str, Any]:
        """Perform initial eligibility checks"""
        eligibility = {
            'age_eligible': True,
            'procedure_covered': 'unknown',
            'waiting_period_satisfied': 'unknown',
            'exclusions_apply': False,
            'notes': []
        }
        
        # Age eligibility
        age = parsed_query.get('age')
        if age:
            if age > 65:
                eligibility['notes'].append(f"Age {age} may have restrictions")
            elif age < 18:
                eligibility['notes'].append(f"Minor age {age} - dependent coverage")
        
        # Procedure coverage check
        procedure = parsed_query.get('procedure')
        if procedure:
            if any(excl for excl in policy_context.get('exclusions', []) if procedure.lower() in excl.lower()):
                eligibility['procedure_covered'] = 'excluded'
                eligibility['exclusions_apply'] = True
            else:
                eligibility['procedure_covered'] = 'likely_covered'
        
        # Waiting period check
        policy_duration = parsed_query.get('policy_duration')
        if policy_duration:
            duration_match = re.search(r'(\d+)\s*(month|year)', str(policy_duration), re.IGNORECASE)
            if duration_match:
                duration = int(duration_match.group(1))
                unit = duration_match.group(2).lower()
                
                if unit == 'month' and duration < 24:
                    eligibility['waiting_period_satisfied'] = 'may_not_be_satisfied'
                    eligibility['notes'].append(f"Policy only {duration} months old")
        
        return eligibility
    
    def _enhance_decision(self, decision_data: Dict, eligibility_check: Dict, 
                         policy_context: Dict) -> Dict[str, Any]:
        """Enhance and validate the decision"""
        enhanced = decision_data.copy()
        
        # Adjust confidence based on available information
        confidence = enhanced.get('confidence', 0.8)
        
        if eligibility_check.get('procedure_covered') == 'unknown':
            confidence *= 0.8
        
        if eligibility_check.get('waiting_period_satisfied') == 'unknown':
            confidence *= 0.9
        
        if not policy_context.get('key_clauses'):
            confidence *= 0.7
        
        enhanced['confidence'] = min(confidence, 1.0)
        
        # Add business rule validations
        if eligibility_check.get('exclusions_apply'):
            enhanced['decision'] = 'rejected'
            enhanced['justification'] += " Procedure appears to be excluded under policy terms."
        
        return enhanced
    
    def _create_detailed_clause_references(self, chunks: List[Dict[str, Any]], 
                                         decision: Dict[str, Any]) -> List[ClauseReference]:
        """Create detailed clause references with page numbers and relevance"""
        clause_refs = []
        
        for i, chunk in enumerate(chunks[:5]):  # Top 5 most relevant
            # Extract page number if available
            page_match = re.search(r'\[Page (\d+)\]', chunk['text'])
            page_number = int(page_match.group(1)) if page_match else None
            
            # Clean text for reference
            clean_text = re.sub(r'\[Page \d+\]', '', chunk['text']).strip()
            
            clause_refs.append(ClauseReference(
                clause_id=chunk['id'],
                content=clean_text[:300] + "..." if len(clean_text) > 300 else clean_text,
                relevance_score=chunk.get('importance_score', chunk['score']),
                page_number=page_number
            ))
        
        return clause_refs
    
    def _extract_exclusions(self, text: str) -> List[str]:
        """Extract exclusions from text"""
        exclusions = []
        
        exclusion_patterns = [
            r'excluded?[^.]*',
            r'not covered[^.]*',
            r'shall not be liable[^.]*'
        ]
        
        for pattern in exclusion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            exclusions.extend(matches)
        
        return exclusions
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract benefits from text"""
        benefits = []
        
        benefit_patterns = [
            r'covered?[^.]*',
            r'benefits?[^.]*',
            r'will pay[^.]*'
        ]
        
        for pattern in benefit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            benefits.extend(matches)
        
        return benefits
    
    def _extract_financial_limits(self, text: str) -> List[str]:
        """Extract financial limits and amounts"""
        limits = []
        
        # Amount patterns
        amount_patterns = [
            r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?',
            r'\d+%\s*(?:of|co-payment)',
            r'up\s+to\s+(?:₹|Rs\.?|INR)?\s*[\d,]+'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            limits.extend(matches)
        
        return limits
    
    def _create_safe_fallback_decision(self, query: str, chunks: List[Dict[str, Any]], 
                                     parsed_query: Dict) -> DecisionResponse:
        """Create a safe fallback decision when AI processing fails"""
        return DecisionResponse(
            decision="pending_documents",
            amount=None,
            confidence=0.3,
            justification=f"Unable to process query '{query}' automatically. Manual review required due to system limitations. Please provide additional documentation for proper assessment.",
            referenced_clauses=[
                ClauseReference(
                    clause_id=chunk['id'],
                    content=chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    relevance_score=chunk.get('score', 0.5)
                ) for chunk in chunks[:3]
            ]
        )
