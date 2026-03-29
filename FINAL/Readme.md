# Legal Document Summarizer API

A Flask-based web API that provides legal document analysis and summarization using machine learning.

## Features

- **Document Upload**: Support for PDF and text files
- **Multiple Summarization Methods**:
  - Extractive (TF-IDF based)
  - Abstractive (BART transformer model)
  - Hybrid (combination approach)
- **Legal Analysis**: Risk assessment, jurisdiction extraction, key terms identification
- **RESTful API**: Clean JSON responses with CORS support

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Start the API Server
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```
GET /health
```

#### 2. Analyze Document (File Upload)
```
POST /analyze
Content-Type: multipart/form-data

Form Data:
- file: PDF or TXT file to analyze
```

#### 3. Analyze Text (Direct Text Input)
```
POST /analyze-text
Content-Type: application/json

{
    "text": "Legal document text content",
    "document_name": "Optional document name"
}
```

#### 4. Model Information
```
GET /models/info
```

### Response Format

```json
{
    "success": true,
    "document_name": "document.pdf",
    "analysis": {
        "summary": "Generated summary text...",
        "risk_level": "Medium",
        "jurisdiction": "New York, USA",
        "termination_terms": "30 days notice",
        "key_terms": ["contract", "liability", "breach"],
        "statistics": {
            "original_words": 1500,
            "summary_words": 200,
            "compression_ratio": "13.3%"
        }
    },
    "timestamp": "2026-03-29T21:49:00.000Z"
}
```

## Integration with HTML Frontend

To integrate with your HTML frontend:

1. Update the JavaScript in `index(2).html` to call the API:

```javascript
async function processFile(input) {
    if (!input.files || !input.files[0]) return;
    
    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update your UI with the analysis results
            openSummary(result.document_name, result.analysis.summary, 
                       `Risk Level: ${result.analysis.risk_level}`, 
                       result.analysis.jurisdiction, 
                       result.analysis.termination_terms);
        } else {
            alert('Analysis failed: ' + result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}
```

## File Structure

```
├── app.py                      # Flask API server
├── config.py                   # Configuration settings
├── legal_summarizer.py         # Main summarizer class
├── document_loader.py          # Document loading utilities
├── text_preprocessor.py        # Text preprocessing
├── extractive_summarizer.py    # Extractive summarization
├── abstractive_summarizer.py   # Abstractive summarization
├── hybrid_summarizer.py        # Hybrid approach
├── requirements.txt           # Python dependencies
├── index(2).html              # Frontend HTML
├── script(3).js               # Frontend JavaScript
├── style.css                  # Frontend styles
└── outputs/                   # Output directory
```

## Configuration

Edit `config.py` to modify:
- Model settings
- API host and port
- File size limits
- Output directories

## Notes

- The API supports files up to 50MB
- First-time model download may take a few minutes
- GPU acceleration is automatically used if available
- All uploaded files are temporarily stored and deleted after processing
