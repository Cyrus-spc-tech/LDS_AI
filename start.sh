#!/bin/bash
# Render startup script
echo "Starting Legal Document Analyzer..."
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

# Install dependencies
pip install -r requirements.txt

# Start the app
gunicorn app_minimal:app --bind 0.0.0.0:$PORT --workers 1
