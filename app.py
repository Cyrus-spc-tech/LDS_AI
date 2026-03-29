from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename

from config import config
from legal_summarizer import LegalSummarizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure app
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = config.UPLOAD_DIR

# Initialize summarizer
summarizer = LegalSummarizer(config)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Legal Document Summarizer API',
        'version': '1.0.0'
    })


@app.route('/analyze', methods=['POST'])
def analyze_document():
    """
    Analyze uploaded legal document
    
    Expected form data:
    - file: The document file (PDF, TXT)
    """
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
        
        # Process document
        print(f"Processing document: {file_path}")
        results = summarizer.process_document(file_path)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass
        
        # Check for processing errors
        if 'error' in results:
            return jsonify({
                'error': 'Document processing failed',
                'details': results['error']
            }), 500
        
        # Format response for frontend
        response = {
            'success': True,
            'document_name': filename,
            'analysis': {
                'summary': results['recommended_summary'],
                'risk_level': results['legal_analysis']['risk_level'],
                'jurisdiction': results['legal_analysis']['jurisdiction'],
                'termination_terms': results['legal_analysis']['termination_terms'],
                'key_terms': results['key_terms'][:10],  # Top 10 terms
                'statistics': {
                    'original_words': results['statistics']['original_words'],
                    'summary_words': results['statistics']['hybrid_words'],
                    'compression_ratio': f"{results['statistics']['hybrid_compression']:.1%}"
                },
                'all_summaries': {
                    'extractive': results['summaries']['extractive'],
                    'abstractive': results['summaries']['abstractive'],
                    'hybrid': results['summaries']['hybrid']
                }
            },
            'timestamp': results['timestamp']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """
    Analyze text content directly
    
    Expected JSON payload:
    {
        "text": "Legal document text content",
        "document_name": "Optional document name"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        document_name = data.get('document_name', 'Text Input')
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Create temporary file for processing
        temp_filename = f"temp_{uuid.uuid4()}.txt"
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Process document
        print(f"Processing text input: {document_name}")
        results = summarizer.process_document(temp_file_path)
        
        # Clean up temporary file
        try:
            os.remove(temp_file_path)
        except OSError:
            pass
        
        # Check for processing errors
        if 'error' in results:
            return jsonify({
                'error': 'Text processing failed',
                'details': results['error']
            }), 500
        
        # Format response
        response = {
            'success': True,
            'document_name': document_name,
            'analysis': {
                'summary': results['recommended_summary'],
                'risk_level': results['legal_analysis']['risk_level'],
                'jurisdiction': results['legal_analysis']['jurisdiction'],
                'termination_terms': results['legal_analysis']['termination_terms'],
                'key_terms': results['key_terms'][:10],
                'statistics': {
                    'original_words': results['statistics']['original_words'],
                    'summary_words': results['statistics']['hybrid_words'],
                    'compression_ratio': f"{results['statistics']['hybrid_compression']:.1%}"
                }
            },
            'timestamp': results['timestamp']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
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
            'abstractive': {
                'name': config.ABSTRACTIVE_MODEL,
                'description': 'Neural abstractive summarization using transformers'
            },
            'hybrid': {
                'name': 'Hybrid Approach',
                'description': 'Combines extractive and abstractive methods for best results'
            }
        },
        'recommended': 'hybrid'
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print(f"Starting Legal Document Summarizer API on {config.API_HOST}:{config.API_PORT}")
    print(f"Upload directory: {config.UPLOAD_DIR}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )
