#!/bin/bash
echo "🚀 Starting FastAPI Voice Assistant Backend..."

# Default PORT if Azure doesn't set it
export PORT=${PORT:-8000}

# Install dependencies
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Start FastAPI with uvicorn  
echo "🌐 Running server on port $PORT..."
python -m uvicorn main:app --host 0.0.0.0 --port $PORT