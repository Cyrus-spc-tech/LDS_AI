import json
from typing import Dict, Any, List
from datetime import datetime

from config import Config
from document_loader import DocumentLoader
from text_preprocessor import TextPreprocessor
from extractive_summarizer import ExtractiveSummarizer
from abstractive_summarizer import AbstractiveSummarizer
from hybrid_summarizer import HybridSummarizer


class LegalSummarizer:
    """Main class for legal document summarization"""
    
    def __init__(self, config: Config = None):
        """Initialize the legal summarizer with configuration"""
        self.config = config or Config()
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.extractive_summarizer = ExtractiveSummarizer(max_features=self.config.TFIDF_MAX_FEATURES)
        self.abstractive_summarizer = AbstractiveSummarizer(model_name=self.config.ABSTRACTIVE_MODEL)
        self.hybrid_summarizer = HybridSummarizer(self.config)
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a legal document and generate summaries
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing summaries and metadata
        """
        try:
            # Load document
            raw_text = self.document_loader.load_document(file_path)
            
            # Preprocess text
            cleaned_text, sentences = self.preprocessor.preprocess(raw_text)
            
            # Generate summaries
            extractive_summary = self.extractive_summarizer.summarize(
                sentences,
                num_sentences=self.config.EXTRACTIVE_SENTENCES
            )
            
            abstractive_summary = self.abstractive_summarizer.summarize(
                cleaned_text,
                max_length=self.config.MAX_SUMMARY_LENGTH,
                min_length=self.config.MIN_SUMMARY_LENGTH
            )
            
            hybrid_summary = self.hybrid_summarizer.summarize(cleaned_text, sentences)
            
            # Extract key terms
            key_terms = self.preprocessor.extract_key_terms(cleaned_text)
            
            # Calculate statistics
            original_words = len(cleaned_text.split())
            extractive_words = len(extractive_summary.split())
            abstractive_words = len(abstractive_summary.split())
            hybrid_words = len(hybrid_summary.split())
            
            # Determine risk level (simple heuristic)
            risk_level = self._assess_risk_level(cleaned_text, key_terms)
            
            # Determine jurisdiction (simple heuristic)
            jurisdiction = self._extract_jurisdiction(cleaned_text)
            
            # Determine termination clause (simple heuristic)
            termination = self._extract_termination_terms(cleaned_text)
            
            result = {
                'file_path': file_path,
                'timestamp': datetime.now().isoformat(),
                'original_text_length': len(cleaned_text),
                'summaries': {
                    'extractive': extractive_summary,
                    'abstractive': abstractive_summary,
                    'hybrid': hybrid_summary
                },
                'key_terms': key_terms,
                'statistics': {
                    'original_words': original_words,
                    'extractive_words': extractive_words,
                    'abstractive_words': abstractive_words,
                    'hybrid_words': hybrid_words,
                    'extractive_compression': extractive_words / original_words if original_words > 0 else 0,
                    'abstractive_compression': abstractive_words / original_words if original_words > 0 else 0,
                    'hybrid_compression': hybrid_words / original_words if original_words > 0 else 0
                },
                'legal_analysis': {
                    'risk_level': risk_level,
                    'jurisdiction': jurisdiction,
                    'termination_terms': termination
                },
                'recommended_summary': hybrid_summary  # Hybrid is typically best
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'file_path': file_path,
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_risk_level(self, text: str, key_terms: List[str]) -> str:
        """Simple heuristic to assess risk level"""
        high_risk_terms = ['breach', 'liability', 'penalty', 'forfeiture', 'termination', 'violation', 'illegal', 'unlawful']
        medium_risk_terms = ['obligation', 'condition', 'limited', 'restricted', 'compliance']
        
        text_lower = text.lower()
        high_risk_count = sum(1 for term in high_risk_terms if term in text_lower)
        medium_risk_count = sum(1 for term in medium_risk_terms if term in text_lower)
        
        if high_risk_count >= 3:
            return "High"
        elif high_risk_count >= 1 or medium_risk_count >= 2:
            return "Medium"
        else:
            return "Low"
    
    def _extract_jurisdiction(self, text: str) -> str:
        """Extract jurisdiction information"""
        jurisdictions = [
            'New York', 'California', 'Texas', 'Florida', 'Illinois', 'Pennsylvania',
            'Federal', 'State', 'Delaware', 'Nevada', 'Washington', 'Massachusetts'
        ]
        
        text_lower = text.lower()
        for jurisdiction in jurisdictions:
            if jurisdiction.lower() in text_lower:
                return jurisdiction + ", USA"
        
        return "Not specified"
    
    def _extract_termination_terms(self, text: str) -> str:
        """Extract termination terms"""
        termination_patterns = [
            r'(\d+)\s*days?\s*notice',
            r'immediate',
            r'(\d+)\s*days?\s*after',
            r'upon.*breach'
        ]
        
        import re
        for pattern in termination_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "Standard terms"
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config.OUTPUT_DIR}/summary_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return output_path
