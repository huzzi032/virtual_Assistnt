# server/tools/zoom_webhook_tool.py
"""
Zoom Webhook Handler
Listens for Zoom meeting events: started, ended, live transcription
"""

import json
import hmac
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import Request, HTTPException
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

class ZoomWebhookHandler:
    def __init__(self):
        """Initialize Zoom webhook handler"""
        
        # Zoom webhook configuration
        self.webhook_secret = os.getenv('ZOOM_SECRET_TOKEN')
        self.verification_token = os.getenv('ZOOM_SECRET_TOKEN')  # Use same secret for verification
        
        # Database for storing webhook events
        self.db_path = "database.db"
        self._init_database()
        
        print(f"🎣 Zoom Webhook Handler initialized")
        print(f"   Secret configured: {'✅' if self.webhook_secret else '❌ Missing ZOOM_SECRET_TOKEN'}")
        print(f"   Verification token: {'✅' if self.verification_token else '❌ Missing ZOOM_SECRET_TOKEN'}")

    def _init_database(self):
        """Initialize database table for webhook events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Create zoom_webhook_events table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zoom_webhook_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    meeting_id TEXT,
                    meeting_uuid TEXT,
                    meeting_topic TEXT,
                    event_timestamp DATETIME,
                    payload TEXT,
                    processed BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create zoom_live_transcripts table for real-time transcription
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zoom_live_transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL,
                    meeting_uuid TEXT,
                    speaker_name TEXT,
                    transcript_content TEXT,
                    timestamp DATETIME,
                    sequence_number INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create zoom_meeting_sessions table for tracking active sessions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zoom_meeting_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT UNIQUE NOT NULL,
                    meeting_uuid TEXT,
                    topic TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    status TEXT DEFAULT 'active',
                    real_time_processing BOOLEAN DEFAULT 0,
                    auto_summary BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            print("✅ Zoom webhook database tables initialized")
            
        except Exception as e:
            print(f"❌ Webhook database initialization error: {e}")

    def verify_webhook_signature(self, payload_body: bytes, signature: str) -> bool:
        """Verify Zoom webhook signature for security"""
        
        if not self.webhook_secret:
            print("⚠️ Webhook secret not configured - skipping signature verification")
            return True  # Allow in development
        
        try:
            # Calculate expected signature
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload_body,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            print(f"❌ Signature verification error: {e}")
            return False

    async def handle_webhook_event(self, request: Request) -> dict:
        """Handle incoming Zoom webhook event with improved error logging"""
        try:
            # Get request body and headers
            body = await request.body()
            headers = dict(request.headers)
            print(f"🔔 Incoming Zoom webhook: {len(body)} bytes")
            print(f"Headers: {headers}")
            try:
                payload = json.loads(body.decode())
                print(f"Payload: {json.dumps(payload, indent=2)}")
            except Exception as e:
                print(f"❌ Invalid JSON payload: {e}")
                return {"success": False, "error": "Invalid JSON payload"}

            # Verify signature if provided
            signature = headers.get('authorization')
            if signature and not self.verify_webhook_signature(body, signature):
                print("❌ Invalid webhook signature")
                return {"success": False, "error": "Invalid webhook signature"}

            # Check for required event field
            event_type = payload.get('event')
            if not event_type:
                print("❌ Missing 'event' field in payload")
                return {"success": False, "error": "Missing 'event' field in payload"}

            # Handle different event types
            if event_type == 'endpoint.url_validation':
                return await self._handle_url_validation(payload)
            elif event_type in ['meeting.started', 'meeting.ended', 'meeting.participant_joined']:
                return await self._handle_meeting_event(payload)
            elif event_type == 'recording.transcript_completed':
                return await self._handle_transcript_completed(payload)
            elif event_type == 'meeting.live_transcript_update':
                return await self._handle_live_transcript(payload)
            else:
                await self._log_unknown_event(payload)
                print(f"❓ Unknown event type: {event_type}")
                return {"success": False, "error": f"Unknown event type: {event_type}"}
        except Exception as e:
            print(f"❌ Webhook handling error: {e}")
            return {"success": False, "error": f"Webhook processing error: {str(e)}"}

    async def _handle_url_validation(self, payload: dict) -> dict:
        """Handle Zoom URL validation challenge"""
        
        # Extract challenge information
        plain_token = payload.get('payload', {}).get('plainToken')
        
        if not plain_token:
            raise HTTPException(status_code=400, detail="Missing plain token for validation")
        
        if not self.verification_token:
            raise HTTPException(status_code=500, detail="Verification token not configured")
        
        # Create challenge response
        challenge_response = {
            'plainToken': plain_token,
            'encryptedToken': hmac.new(
                self.verification_token.encode(),
                plain_token.encode(),
                hashlib.sha256
            ).hexdigest()
        }
        
        print(f"✅ Zoom URL validation successful")
        return challenge_response

    async def _handle_meeting_event(self, payload: dict) -> dict:
        """Handle meeting started/ended events"""
        
        event_type = payload.get('event')
        meeting_data = payload.get('payload', {}).get('object', {})
        
        meeting_id = meeting_data.get('id')
        meeting_uuid = meeting_data.get('uuid')
        topic = meeting_data.get('topic', 'Untitled Meeting')
        start_time = meeting_data.get('start_time')
        
        print(f"📅 Meeting event: {event_type}")
        print(f"   Meeting ID: {meeting_id}")
        print(f"   Topic: {topic}")
        
        try:
            # Store event in database
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Log the webhook event
            cur.execute("""
                INSERT INTO zoom_webhook_events 
                (event_type, meeting_id, meeting_uuid, meeting_topic, event_timestamp, payload)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_type,
                meeting_id,
                meeting_uuid,
                topic,
                datetime.now(),
                json.dumps(payload)
            ))
            
            # Handle specific event types
            if event_type == 'meeting.started':
                await self._handle_meeting_started(cur, meeting_id, meeting_uuid, topic, start_time)
            elif event_type == 'meeting.ended':
                await self._handle_meeting_ended(cur, meeting_id, meeting_uuid)
            
            conn.commit()
            conn.close()
            
            return {"success": True, "status": "success", "event_type": event_type, "meeting_id": meeting_id}
            
        except Exception as e:
            print(f"❌ Database error handling meeting event: {e}")
            return {"success": False, "status": "error", "error": str(e)}

    async def _handle_meeting_started(self, cur, meeting_id: str, meeting_uuid: str, topic: str, start_time: str):
        """Handle meeting started event"""
        
        # Insert or update meeting session
        cur.execute("""
            INSERT OR REPLACE INTO zoom_meeting_sessions
            (meeting_id, meeting_uuid, topic, start_time, status, real_time_processing)
            VALUES (?, ?, ?, ?, 'active', 1)
        """, (meeting_id, meeting_uuid, topic, start_time or datetime.now()))
        
        print(f"🟢 Meeting started: {topic} (ID: {meeting_id})")
        
        # TODO: Trigger real-time processing setup
        # This could include:
        # - Setting up real-time transcript listeners
        # - Initializing AI processing pipeline
        # - Notifying connected clients

    async def _handle_meeting_ended(self, cur, meeting_id: str, meeting_uuid: str):
        """Handle meeting ended event"""
        
        # Update meeting session status
        cur.execute("""
            UPDATE zoom_meeting_sessions 
            SET status = 'ended', end_time = ?, updated_at = ?
            WHERE meeting_id = ?
        """, (datetime.now(), datetime.now(), meeting_id))
        
        print(f"🔴 Meeting ended: {meeting_id}")
        
        # TODO: Trigger post-meeting processing
        # This could include:
        # - Generating meeting summary
        # - Processing accumulated transcripts
        # - Extracting action items and todos
        # - Sending summary email

    async def _handle_live_transcript(self, payload: dict) -> dict:
        """Handle real-time transcript updates"""
        
        transcript_data = payload.get('payload', {})
        meeting_id = transcript_data.get('meeting_id')
        meeting_uuid = transcript_data.get('meeting_uuid')
        
        # Extract transcript content
        transcript_content = transcript_data.get('transcript_content', '')
        speaker_name = transcript_data.get('speaker_name', 'Unknown')
        timestamp = transcript_data.get('timestamp')
        sequence_number = transcript_data.get('sequence_number', 0)
        
        if not transcript_content.strip():
            return {"success": True, "status": "empty_transcript"}
        
        print(f"💬 Live transcript: {speaker_name}: {transcript_content[:50]}...")
        
        try:
            # Store live transcript
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO zoom_live_transcripts
                (meeting_id, meeting_uuid, speaker_name, transcript_content, timestamp, sequence_number)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                meeting_id,
                meeting_uuid, 
                speaker_name,
                transcript_content,
                timestamp or datetime.now(),
                sequence_number
            ))
            
            conn.commit()
            conn.close()
            
            # TODO: Trigger real-time AI processing
            # This could include:
            # - Real-time action item detection
            # - Live todo extraction
            # - Sentiment analysis
            # - Key topic identification
            
            return {
                "status": "success",
                "meeting_id": meeting_id,
                "speaker": speaker_name,
                "transcript_length": len(transcript_content)
            }
            
        except Exception as e:
            print(f"❌ Error storing live transcript: {e}")
            return {"success": False, "status": "error", "error": str(e)}

    async def _handle_transcript_completed(self, payload: dict) -> dict:
        """Handle completed transcript notification"""
        
        recording_data = payload.get('payload', {}).get('object', {})
        meeting_id = recording_data.get('meeting_id')
        download_url = recording_data.get('recording_files', [{}])[0].get('download_url')
        
        print(f"📝 Transcript completed for meeting: {meeting_id}")
        print(f"   Download URL available: {'✅' if download_url else '❌'}")
        
        # TODO: Download and process completed transcript
        # This could include:
        # - Downloading the full transcript file
        # - Running comprehensive AI analysis
        # - Generating detailed meeting summary
        # - Extracting all action items and decisions
        
        return {"success": True, "status": "transcript_noted", "meeting_id": meeting_id}

    async def _log_unknown_event(self, payload: dict):
        """Log unknown webhook events for debugging"""
        
        event_type = payload.get('event', 'unknown')
        
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO zoom_webhook_events 
                (event_type, payload, event_timestamp)
                VALUES (?, ?, ?)
            """, (f"unknown_{event_type}", json.dumps(payload), datetime.now()))
            
            conn.commit()
            conn.close()
            
            print(f"❓ Unknown Zoom event logged: {event_type}")
            
        except Exception as e:
            print(f"❌ Error logging unknown event: {e}")

    def get_active_meetings(self) -> list:
        """Get list of currently active meetings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT meeting_id, meeting_uuid, topic, start_time, real_time_processing
                FROM zoom_meeting_sessions
                WHERE status = 'active'
                ORDER BY start_time DESC
            """)
            
            results = cur.fetchall()
            conn.close()
            
            active_meetings = []
            for row in results:
                active_meetings.append({
                    "meeting_id": row[0],
                    "meeting_uuid": row[1],
                    "topic": row[2],
                    "start_time": row[3],
                    "real_time_processing": bool(row[4])
                })
            
            return active_meetings
            
        except Exception as e:
            print(f"❌ Error fetching active meetings: {e}")
            return []

    def get_meeting_transcript(self, meeting_id: str) -> list:
        """Get accumulated transcript for a meeting"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT speaker_name, transcript_content, timestamp, sequence_number
                FROM zoom_live_transcripts
                WHERE meeting_id = ?
                ORDER BY sequence_number ASC, timestamp ASC
            """, (meeting_id,))
            
            results = cur.fetchall()
            conn.close()
            
            transcript_segments = []
            for row in results:
                transcript_segments.append({
                    "speaker": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "sequence": row[3]
                })
            
            return transcript_segments
            
        except Exception as e:
            print(f"❌ Error fetching meeting transcript: {e}")
            return []

# Global instance
zoom_webhook_handler = ZoomWebhookHandler()

# Convenience functions
async def handle_zoom_webhook(request: Request) -> dict:
    """Handle incoming Zoom webhook"""
    return await zoom_webhook_handler.handle_webhook_event(request)

def get_active_zoom_meetings() -> list:
    """Get active Zoom meetings"""
    return zoom_webhook_handler.get_active_meetings()

def get_zoom_meeting_transcript(meeting_id: str) -> list:
    """Get transcript for Zoom meeting"""
    return zoom_webhook_handler.get_meeting_transcript(meeting_id)

if __name__ == "__main__":
    # Test webhook handler
    handler = ZoomWebhookHandler()
    
    # Test database
    active = handler.get_active_meetings()
    print(f"Active meetings: {active}")