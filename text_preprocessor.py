import re
from typing import List, Tuple

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class TextPreprocessor:
    """Handles text cleaning and preprocessing for legal documents"""
    
    def __init__(self):
        """Initialize spaCy model for sentence segmentation"""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception as e:
                    print(f"Failed to download spaCy model: {e}")
                    print("Using basic sentence splitting instead")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove legal citation patterns
        text = re.sub(r'\[\d{4}\]\s+[A-Z]+\s+\d+', '', text)
        text = re.sub(r'\(\d{4}\)\s+\d+', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove page numbers
        text = re.sub(r'Page\s+\d+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if not text:
            return []
        
        if self.nlp:
            # Use spaCy for better sentence segmentation
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to basic sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        # Filter out very short sentences
        return [sent for sent in sentences if len(sent) > 10]
    
    def preprocess(self, text: str) -> Tuple[str, List[str]]:
        """Main preprocessing pipeline"""
        if not text:
            return "", []
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Segment into sentences
        sentences = self.segment_sentences(cleaned_text)
        
        return cleaned_text, sentences
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key legal terms from text"""
        if not text:
            return []
        
        # Common legal terms pattern
        legal_terms_pattern = r'\b(?:contract|agreement|liability|indemnity|jurisdiction|termination|breach|warranty|representation|covenant|provision|clause|section|article|statute|regulation|compliance|obligation|right|remedy|damages|penalty|forfeiture|arbitration|mediation|litigation|court|judge|plaintiff|defendant|evidence|testimony|affidavit|deposition|subpoena|injunction|restraining|preliminary|permanent|appeal|reversal|affirmance|precedent|case law|common law|civil law|criminal law|constitutional|federal|state|local|municipal|ordinance|statute|act|code|rule|regulation|policy|procedure|guideline|standard|requirement|specification|criteria|condition|stipulation|qualification|exception|exemption|waiver|release|disclaimer|confidentiality|non-disclosure|proprietary|trade secret|intellectual property|patent|trademark|copyright|license|royalty|infringement|validity|enforceability|void|voidable|unenforceable|illegal|unlawful|prohibited|restricted|limited|conditional|subject to|contingent upon|provided that|notwithstanding|however|therefore|whereas|pursuant to|in accordance with|compliance with|adherence to|violation of|breach of|failure to|default in|non-performance of|misrepresentation of|fraudulent|negligent|intentional|willful|malicious|gross|ordinary|reasonable|good faith|bad faith|due diligence|best efforts|commercially reasonable|material|immaterial|substantial|insignificant|de minimis|temporary|permanent|irrevocable|revocable|terminable|non-terminable|perpetual|limited|unlimited|exclusive|non-exclusive|transferable|non-transferable|assignable|non-assignable|sublicensable|non-sublicensable)\b'
        
        # Find all legal terms
        terms = re.findall(legal_terms_pattern, text, re.IGNORECASE)
        
        # Return unique terms (case-insensitive)
        unique_terms = list(set(term.lower() for term in terms))
        
        return unique_terms[:20]  # Limit to top 20 terms
