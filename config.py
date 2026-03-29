from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Configuration for Legal Document Summarizer"""
    
    # Model Configuration
    ABSTRACTIVE_MODEL: str = "facebook/bart-large-cnn"
    MAX_SUMMARY_LENGTH: int = 750
    MIN_SUMMARY_LENGTH: int = 450
    EXTRACTIVE_SENTENCES: int = 5
    
    # TF-IDF Configuration
    TFIDF_MAX_FEATURES: int = 5000
    
    # File Paths
    SAMPLE_DOCUMENT_PATH: str = "sample_legal_document.txt"
    OUTPUT_DIR: str = "outputs"
    UPLOAD_DIR: str = "uploads"
    
    # Text Processing
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5000
    DEBUG: bool = True
    
    def __post_init__(self):
        """Create necessary directories after initialization"""
        Path(self.OUTPUT_DIR).mkdir(exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(exist_ok=True)

# Global configuration instance
config = Config()
