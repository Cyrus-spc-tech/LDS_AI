from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import re
from werkzeug.utils import secure_filename
import PyPDF2

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

class MinimalLegalAnalyzer:
    """Ultra-minimal legal document analyzer using only built-in libraries"""
    
    def __init__(self):
        pass
    
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
        # Remove basic patterns
        text = re.sub(r'\[\d{4}\]\s+[A-Z]+\s+\d+', '', text)
        text = re.sub(r'\(\d{4}\)\s+\d+', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'Page\s+\d+', '', text)
        return text.strip()
    
    def simple_summarize(self, text, max_sentences=3):
        """Simple extractive summarization using sentence length and position"""
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'
        
        # Score sentences by length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Prefer sentences in the middle of the document
            position_score = 1.0 - abs(i - len(sentences)/2) / (len(sentences)/2)
            # Prefer medium-length sentences
            length_score = min(len(sentence), 100) / 100
            # Combined score
            score = position_score * 0.6 + length_score * 0.4
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
        
        # Maintain original order
        result_sentences = []
        for sentence in sentences:
            if sentence in top_sentences and sentence not in result_sentences:
                result_sentences.append(sentence)
        
        return '. '.join(result_sentences) + '.'
    
    def analyze_risk(self, text):
        """Basic risk analysis using keyword detection"""
        risk_keywords = {
            'high': ['terminate', 'breach', 'liability', 'penalty', 'violation', 'lawsuit', 'void', 'illegal'],
            'medium': ['obligation', 'responsibility', 'compliance', 'regulation', 'deadline', 'requirement'],
            'low': ['agreement', 'cooperation', 'partnership', 'mutual', 'beneficial', 'support']
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
        """Extract key terms using simple frequency analysis"""
        # Simple word frequency analysis
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Common legal and business words to filter out
        stop_words = {
            'that', 'this', 'with', 'from', 'they', 'have', 'been', 'said', 'each', 'which',
            'their', 'time', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'shall', 'can', 'into', 'upon', 'such', 'more', 'most', 'some', 'any', 'what',
            'when', 'where', 'why', 'how', 'only', 'very', 'also', 'even', 'well', 'much',
            'many', 'still', 'being', 'does', 'did', 'than', 'make', 'like', 'just', 'know',
            'take', 'come', 'made', 'find', 'where', 'much', 'through', 'between', 'both',
            'after', 'before', 'here', 'there', 'another', 'other', 'those', 'these', 'same',
            'over', 'under', 'again', 'further', 'then', 'once', 'work', 'call', 'try', 'ask',
            'need', 'feel', 'seem', 'leave', 'put', 'keep', 'let', 'begin', 'seem', 'write',
            'give', 'become', 'turn', 'open', 'walk', 'win', 'offer', 'believe', 'hold',
            'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet',
            'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch',
            'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow',
            'open', 'walk', 'win', 'offer', 'believe', 'hold', 'bring', 'happen', 'write',
            'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set',
            'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create',
            'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer',
            'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose',
            'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
            'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow'
        }
        
        # Filter out stop words and count frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top terms
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_terms]
        return [term for term, freq in top_terms]
    
    def process_document(self, text):
        """Process document and return analysis"""
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Generate summary
            summary = self.simple_summarize(cleaned_text)
            
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
analyzer = MinimalLegalAnalyzer()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Legal Document Analyzer API (Minimal)',
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
                'name': 'Simple Extractive Summarizer',
                'description': 'Selects important sentences based on position and length'
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
    print("Starting Legal Document Analyzer API (Minimal)")
    print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    
    # Always use the port from environment variable
    port = int(os.environ.get('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting on port {port} with debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
