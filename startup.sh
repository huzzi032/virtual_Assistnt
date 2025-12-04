#!/bin/bash

# Voice Assistant Backend Startup Script for Azure
echo "🚀 Starting Voice Assistant Backend..."

# Set default port if not provided by Azure
export PORT=${PORT:-8000}

# Install dependencies if requirements changed
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Initialize database tables
echo "🗄️ Initializing database..."
python -c "
import sqlite3
import os

# Create database and tables
conn = sqlite3.connect('database.db')
cur = conn.cursor()

# Create todos table
cur.execute('''CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

# Create voice_inputs table  
cur.execute('''CREATE TABLE IF NOT EXISTS voice_inputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

# Create zoom_tokens table
cur.execute('''CREATE TABLE IF NOT EXISTS zoom_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE,
    access_token TEXT,
    refresh_token TEXT,
    expires_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

# Create zoom_webhook_events table
cur.execute('''CREATE TABLE IF NOT EXISTS zoom_webhook_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    meeting_id TEXT,
    meeting_uuid TEXT,
    meeting_topic TEXT,
    event_timestamp DATETIME,
    payload TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

# Create zoom_meeting_sessions table
cur.execute('''CREATE TABLE IF NOT EXISTS zoom_meeting_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT,
    meeting_uuid TEXT,
    topic TEXT,
    start_time DATETIME,
    end_time DATETIME,
    duration INTEGER,
    participant_count INTEGER DEFAULT 0,
    recording_started BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

conn.commit()
conn.close()
print('✅ Database initialized')
"

# Start the application
echo "🌐 Starting server on port $PORT..."
echo "📍 Server will be available at: https://virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net"
echo "🔗 Zoom OAuth Callback: https://virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net/api/zoom/auth/callback"
echo "🎣 Zoom Webhook Endpoint: https://virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net/webhooks/zoom"

# Run the main application
python main.py