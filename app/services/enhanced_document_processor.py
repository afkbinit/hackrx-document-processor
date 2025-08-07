# app/services/enhanced_document_processor.py
import requests
import PyPDF2
from docx import Document
import re
from typing import List, Dict, Any
import io
from app.utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedDocumentProcessor:
    def __init__(self):
        # Insurance-specific patterns
        self.clause_patterns = [
            r'(?:clause|section|article|paragraph)\s*\d+(?:\.\d+)*',
            r'waiting\s*period',
            r'pre[-\s]existing',
            r'co[-\s]payment',
            r'sum\s+insured',
            r'deductible',
            r'exclusion',
            r'maternity',
            r'dental\s+treatment',
            r'AYUSH\s+treatment'
        ]
        
        self.benefit_patterns = [
            r'room\s+rent',
            r'ICU\s+charges',
            r'ambulance',
            r'day\s+care',
            r'hospitalization',
            r'pre[-\s]hospitalization',
            r'post[-\s]hospitalization'
        ]
    
    async def process_document(self, document_url: str) -> Dict[str, Any]:
        """Enhanced document processing with insurance focus"""
        try:
            response = requests.get(document_url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type:
                text_content = self._extract_pdf_text_enhanced(response.content)
            elif 'word' in content_type or 'docx' in content_type:
                text_content = self._extract_docx_text_enhanced(response.content)
            else:
                text_content = response.text
            
            # Create insurance-specific chunks
            chunks = self._create_insurance_chunks(text_content)
            
            # Extract key policy information
            policy_info = self._extract_policy_info(text_content)
            
            return {
                'url': document_url,
                'content_type': content_type,
                'text': text_content,
                'chunks': chunks,
                'total_chunks': len(chunks),
                'policy_info': policy_info
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    def _extract_pdf_text_enhanced(self, content: bytes) -> str:
        """Enhanced PDF extraction with better formatting"""
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                # Clean up the text
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', page_text)
                text += f"[Page {page_num + 1}] {page_text}\n"
                
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            
        return text
    
    def _create_insurance_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Create chunks optimized for insurance content"""
        chunks = []
        
        # Split by sections first (clauses, benefits, etc.)
        section_splits = re.split(r'(?:SECTION|CLAUSE|BENEFIT|COVERAGE|EXCLUSION)\s*[A-Z0-9]', text, flags=re.IGNORECASE)
        
        chunk_id = 0
        for section_idx, section in enumerate(section_splits):
            if not section.strip():
                continue
                
            words = section.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Identify chunk type and importance
                chunk_type = self._identify_chunk_type(chunk_text)
                importance_score = self._calculate_importance_score(chunk_text)
                
                # Extract specific insurance terms
                insurance_terms = self._extract_insurance_terms(chunk_text)
                
                chunks.append({
                    'id': f"chunk_{chunk_id}",
                    'text': chunk_text,
                    'section_idx': section_idx,
                    'start_idx': i,
                    'end_idx': min(i + chunk_size, len(words)),
                    'word_count': len(chunk_words),
                    'chunk_type': chunk_type,
                    'importance_score': importance_score,
                    'insurance_terms': insurance_terms,
                    'clause_references': re.findall(r'(?:clause|section)\s*\d+(?:\.\d+)*', chunk_text, re.IGNORECASE)
                })
                
                chunk_id += 1
        
        # Sort by importance for better retrieval
        chunks.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return chunks
    
    def _identify_chunk_type(self, text: str) -> str:
        """Identify the type of insurance content"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['exclusion', 'not covered', 'excluded']):
            return 'exclusion'
        elif any(term in text_lower for term in ['waiting period', 'wait', 'months']):
            return 'waiting_period'
        elif any(term in text_lower for term in ['benefit', 'coverage', 'covered']):
            return 'benefit'
        elif any(term in text_lower for term in ['procedure', 'surgery', 'treatment']):
            return 'procedure'
        elif any(term in text_lower for term in ['premium', 'sum insured', 'co-payment']):
            return 'financial'
        else:
            return 'general'
    
    def _calculate_importance_score(self, text: str) -> float:
        """Calculate importance score for chunk ranking"""
        score = 0.0
        text_lower = text.lower()
        
        # Higher score for key insurance terms
        key_terms = {
            'waiting period': 3.0,
            'pre-existing': 2.5,
            'exclusion': 2.0,
            'benefit': 1.5,
            'coverage': 1.5,
            'sum insured': 2.0,
            'deductible': 1.5,
            'co-payment': 1.5
        }
        
        for term, weight in key_terms.items():
            if term in text_lower:
                score += weight
        
        # Bonus for numerical information (limits, periods, etc.)
        if re.search(r'\d+\s*(?:days|months|years|%|\$|₹)', text_lower):
            score += 1.0
        
        return score
    
    def _extract_insurance_terms(self, text: str) -> List[str]:
        """Extract insurance-specific terms from chunk"""
        terms = []
        
        for pattern in self.clause_patterns + self.benefit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        # Extract waiting periods
        waiting_periods = re.findall(r'(\d+)\s*(?:days?|months?|years?)\s*waiting\s*period', text, re.IGNORECASE)
        terms.extend([f"{period} waiting period" for period in waiting_periods])
        
        # Extract financial amounts
        amounts = re.findall(r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?', text)
        terms.extend(amounts)
        
        return list(set(terms))  # Remove duplicates
    
    def _extract_policy_info(self, text: str) -> Dict[str, Any]:
        """Extract key policy information"""
        policy_info = {}
        
        # Extract policy number
        policy_num = re.search(r'(?:policy|UIN)[\s:]*([A-Z0-9]+)', text, re.IGNORECASE)
        if policy_num:
            policy_info['policy_number'] = policy_num.group(1)
        
        # Extract waiting periods
        waiting_periods = re.findall(r'(\d+)\s*(days?|months?|years?)\s*waiting\s*period', text, re.IGNORECASE)
        policy_info['waiting_periods'] = waiting_periods
        
        # Extract sum insured ranges
        sum_insured = re.findall(r'(?:sum\s*insured|coverage)[\s:]*(?:₹|Rs\.?|INR)?\s*([\d,]+)', text, re.IGNORECASE)
        policy_info['sum_insured_options'] = sum_insured
        
        return policy_info
