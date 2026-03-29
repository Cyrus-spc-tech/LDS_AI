import torch

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AbstractiveSummarizer:
    """Abstractive summarization using transformer models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """Initialize the abstractive summarization model"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            print("Transformers library not available. Using fallback summarization.")
    
    def _load_model(self):
        """Load the transformer model and tokenizer"""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model = None
            self.tokenizer = None
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Generate abstractive summary
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Generated summary
        """
        if not text:
            return ""
        
        # Check if model is available
        if self.model is None or self.tokenizer is None:
            return self._fallback_summary(text, max_length)
        
        # Check text length
        if len(text.split()) <= 50:
            return text
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3
                )
            
            # Decode summary
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"Abstractive summarization failed: {e}")
            return self._fallback_summary(text, max_length)
    
    def _fallback_summary(self, text: str, max_length: int) -> str:
        """Fallback summarization method"""
        words = text.split()
        if len(words) <= max_length:
            return text
        else:
            return ' '.join(words[:max_length]) + "..."
