import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


class ExtractiveSummarizer:
    """Extractive summarization using TF-IDF scoring"""
    
    def __init__(self, max_features: int = 5000):
        """Initialize TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1,
            max_df=0.9
        )
    
    def summarize(self, sentences: List[str], num_sentences: int = 5) -> str:
        """
        Generate extractive summary by selecting top-scoring sentences
        
        Args:
            sentences: List of sentences from the document
            num_sentences: Number of sentences to include in summary
            
        Returns:
            Extractive summary as a string
        """
        if not sentences:
            return ""
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        try:
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores (sum of TF-IDF values)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Get top sentences indices
            top_indices = np.argsort(sentence_scores)[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            # Extract top sentences
            summary_sentences = [sentences[i] for i in top_indices]
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            print(f"Extractive summarization failed: {e}")
            # Fallback: return first n sentences
            return ' '.join(sentences[:num_sentences])
