from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import re
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import PyPDF2

# Download NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure app
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB for free tier
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

class LightweightLegalAnalyzer:
    """Lightweight legal document analyzer without heavy ML models"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove legal citations (basic pattern)
        text = re.sub(r'\[\d{4}\]\s+[A-Z]+\s+\d+', '', text)
        text = re.sub(r'\(\d{4}\)\s+\d+', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'Page\s+\d+', '', text)
        return text.strip()
    
    def extractive_summarize(self, text, num_sentences=3):
        """Extractive summarization using TF-IDF"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return ' '.join(sentences)
            
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top sentences
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
        except Exception:
            # Fallback to first few sentences
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:num_sentences])
    
    def analyze_risk(self, text):
        """Basic risk analysis using keyword detection"""
        risk_keywords = {
            'high': ['terminate', 'breach', 'liability', 'penalty', 'violation', 'lawsuit'],
            'medium': ['obligation', 'responsibility', 'compliance', 'regulation', 'deadline'],
            'low': ['agreement', 'cooperation', 'partnership', 'mutual', 'beneficial']
        }
        
        text_lower = text.lower()
        risk_scores = {}
        
        for level, keywords in risk_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            risk_scores[level] = score
        
        # Determine risk level
        if risk_scores['high'] > 0:
            return 'high'
        elif risk_scores['medium'] > risk_scores['low']:
            return 'medium'
        else:
            return 'low'
    
    def extract_key_terms(self, text, num_terms=10):
        """Extract key terms using TF-IDF"""
        try:
            # Simple keyword extraction
            words = word_tokenize(text.lower())
            words = [self.stemmer.stem(word) for word in words 
                    if word.isalpha() and word not in self.stop_words]
            
            # Count word frequencies
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top terms
            top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_terms]
            return [term for term, freq in top_terms]
        except Exception:
            return ['legal', 'document', 'analysis']
    
    def process_document(self, text):
        """Process document and return analysis"""
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Generate summary
            summary = self.extractive_summarize(cleaned_text)
            
            # Analyze risk
            risk_level = self.analyze_risk(cleaned_text)
            
            # Extract key terms
            key_terms = self.extract_key_terms(cleaned_text)
            
            # Calculate statistics
            original_words = len(cleaned_text.split())
            summary_words = len(summary.split())
            compression_ratio = (summary_words / original_words * 100) if original_words > 0 else 0
            
            return {
                'success': True,
                'summary': summary,
                'risk_level': risk_level,
                'key_terms': key_terms,
                'statistics': {
                    'original_words': original_words,
                    'summary_words': summary_words,
                    'compression_ratio': f"{compression_ratio:.1f}%"
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Initialize analyzer
analyzer = LightweightLegalAnalyzer()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Legal Document Analyzer API (Lightweight)',
        'version': '1.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Analyze uploaded legal document"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(file_path)
        
        # Extract text from file
        if filename.lower().endswith('.pdf'):
            text = analyzer.extract_text_from_pdf(file_path)
        else:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Process document
        results = analyzer.process_document(text)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass
        
        # Format response
        if results['success']:
            response = {
                'success': True,
                'document_name': filename,
                'analysis': results,
                'timestamp': '2024-01-01T00:00:00Z'
            }
            return jsonify(response)
        else:
            return jsonify({
                'error': 'Document processing failed',
                'details': results['error']
            }), 500
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text content directly"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        document_name = data.get('document_name', 'Text Input')
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Process document
        results = analyzer.process_document(text)
        
        # Format response
        if results['success']:
            response = {
                'success': True,
                'document_name': document_name,
                'analysis': results,
                'timestamp': '2024-01-01T00:00:00Z'
            }
            return jsonify(response)
        else:
            return jsonify({
                'error': 'Text processing failed',
                'details': results['error']
            }), 500
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/models/info', methods=['GET'])
def models_info():
    """Get information about available models"""
    return jsonify({
        'models': {
            'extractive': {
                'name': 'TF-IDF Extractive Summarizer',
                'description': 'Selects important sentences based on TF-IDF scoring'
            },
            'risk_analysis': {
                'name': 'Keyword-based Risk Analysis',
                'description': 'Analyzes document risk using keyword detection'
            }
        },
        'recommended': 'extractive'
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Legal Document Analyzer API (Lightweight)")
    print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    
    # Use Gunicorn for production
    if os.getenv('RENDER') == 'true':
        # Production mode - Render will use Gunicorn
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    else:
        # Development mode
        app.run(host='0.0.0.0', port=5000, debug=True)
