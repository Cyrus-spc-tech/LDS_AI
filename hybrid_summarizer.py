from typing import List
from config import Config
from extractive_summarizer import ExtractiveSummarizer
from abstractive_summarizer import AbstractiveSummarizer


class HybridSummarizer:
    """Hybrid summarization combining extractive and abstractive methods"""
    
    def __init__(self, config: Config):
        """Initialize hybrid summarizer with configuration"""
        self.config = config
        self.extractive = ExtractiveSummarizer(max_features=config.TFIDF_MAX_FEATURES)
        self.abstractive = AbstractiveSummarizer(model_name=config.ABSTRACTIVE_MODEL)
    
    def summarize(self, text: str, sentences: List[str]) -> str:
        """
        Generate hybrid summary by first extracting key sentences,
        then applying abstractive summarization
        
        Args:
            text: Original cleaned text
            sentences: List of sentences from the document
            
        Returns:
            Hybrid summary
        """
        if not text or not sentences:
            return ""
        
        # Step 1: Generate extractive summary
        extractive_summary = self.extractive.summarize(
            sentences,
            num_sentences=self.config.EXTRACTIVE_SENTENCES
        )
        
        # Step 2: Apply abstractive summarization to extractive summary
        hybrid_summary = self.abstractive.summarize(
            extractive_summary,
            max_length=self.config.MAX_SUMMARY_LENGTH,
            min_length=self.config.MIN_SUMMARY_LENGTH
        )
        
        return hybrid_summary
