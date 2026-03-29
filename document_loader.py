from pathlib import Path
from typing import Union

# PDF Libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class DocumentLoader:
    """Handles loading of various document formats"""
    
    @staticmethod
    def load_pdf(file_path: Union[str, Path]) -> str:
        """Load text content from PDF file"""
        file_path = str(file_path)
        
        # Try PyPDF2 first
        if PYPDF2_AVAILABLE:
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                if text.strip():
                    return text
            except Exception as e:
                print(f"PyPDF2 failed: {e}")
        
        # Fallback to pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                if text.strip():
                    return text
            except Exception as e:
                print(f"pdfplumber failed: {e}")
        
        raise FileNotFoundError(f"Cannot read PDF: {file_path}")
    
    @staticmethod
    def load_txt(file_path: Union[str, Path]) -> str:
        """Load text content from text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> str:
        """Load document based on file extension"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return cls.load_pdf(file_path)
        elif suffix == '.txt':
            return cls.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    @classmethod
    def save_uploaded_file(cls, file_content: bytes, filename: str, upload_dir: str) -> str:
        """Save uploaded file content to disk"""
        upload_path = Path(upload_dir) / filename
        upload_path.parent.mkdir(exist_ok=True)
        
        with open(upload_path, 'wb') as f:
            f.write(file_content)
        
        return str(upload_path)
