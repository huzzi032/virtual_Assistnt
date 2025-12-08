"""
Personal Meeting Audio Capture
Captures audio from user's microphone during Zoom meetings
Records from user's side, not as a bot participant
"""

import asyncio
import pyaudio
import wave
import threading
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional
import sqlite3

class PersonalMeetingRecorder:
    def __init__(self):
        """Initialize personal meeting recorder"""
        
        # Audio settings
        self.sample_rate = 16000  # 16kHz for speech
        self.chunk_size = 1024
        self.channels = 1  # Mono
        self.audio_format = pyaudio.paInt16
        
        # Recording state
        self.is_recording = False
        self.current_meeting_id = None
        self.recording_thread = None
        self.audio_buffer = []
        
        # Audio interface
        self.audio = None
        self.stream = None
        
        # Database
        self.db_path = "database.db"
        self._init_database()
        
        print("🎤 Personal Meeting Recorder initialized")
    
    def _init_database(self):
        """Initialize database for personal recordings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Create personal_meeting_recordings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS personal_meeting_recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT NOT NULL,
                    recording_session_id TEXT UNIQUE NOT NULL,
                    start_time DATETIME,
                    end_time DATETIME,
                    audio_file_path TEXT,
                    transcript_file_path TEXT,
                    status TEXT DEFAULT 'active',
                    duration_seconds INTEGER,
                    file_size_bytes INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create personal_transcripts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS personal_transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_session_id TEXT NOT NULL,
                    meeting_id TEXT NOT NULL,
                    transcript_chunk TEXT NOT NULL,
                    timestamp DATETIME,
                    chunk_number INTEGER,
                    confidence_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recording_session_id) REFERENCES personal_meeting_recordings (recording_session_id)
                )
            """)
            
            conn.commit()
            conn.close()
            print("✅ Personal recording database initialized")
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
    
    def start_recording(self, meeting_id: str, meeting_url: Optional[str] = None) -> Dict[str, Any]:
        """Start personal audio recording for a meeting"""
        try:
            if self.is_recording:
                return {
                    "success": False,
                    "error": f"Already recording meeting {self.current_meeting_id}"
                }
            
            print(f"🎤 Starting personal recording for meeting: {meeting_id}")
            
            # Generate unique session ID
            session_id = f"personal_{meeting_id}_{int(time.time())}"
            
            # Create audio directory if not exists
            audio_dir = "recordings"
            os.makedirs(audio_dir, exist_ok=True)
            
            # Audio file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"{audio_dir}/meeting_{meeting_id}_{timestamp}.wav"
            
            # Save recording session to database
            self._save_recording_session(session_id, meeting_id, audio_file)
            
            # Initialize audio recording
            recording_result = self._initialize_audio_recording(audio_file)
            if not recording_result.get('success'):
                return recording_result
            
            # Start recording in background thread
            self.is_recording = True
            self.current_meeting_id = meeting_id
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                args=(session_id, audio_file),
                daemon=True
            )
            self.recording_thread.start()
            
            print(f"✅ Personal recording started: {audio_file}")
            return {
                "success": True,
                "session_id": session_id,
                "meeting_id": meeting_id,
                "audio_file": audio_file,
                "message": f"Personal recording started for meeting {meeting_id}"
            }
            
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    def _initialize_audio_recording(self, audio_file: str) -> Dict[str, Any]:
        """Initialize PyAudio for recording"""
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Check for available audio devices
            device_count = self.audio.get_device_count()
            print(f"🔍 Found {device_count} audio devices")
            
            # Find default input device
            default_device = None
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                max_input_channels = device_info.get('maxInputChannels', 0)
                if isinstance(max_input_channels, (int, float)) and max_input_channels > 0:
                    default_device = i
                    print(f"📱 Using audio device: {device_info['name']}")
                    break
            
            if default_device is None:
                return {"success": False, "error": "No audio input device found"}
            
            # Open audio stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=default_device,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"🎤 Audio stream opened: {self.sample_rate}Hz, {self.channels} channel(s)")
            return {"success": True}
            
        except Exception as e:
            print(f"❌ Audio initialization error: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_recording_session(self, session_id: str, meeting_id: str, audio_file: str):
        """Save recording session to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO personal_meeting_recordings 
                (recording_session_id, meeting_id, start_time, audio_file_path, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                meeting_id,
                datetime.now().isoformat(),
                audio_file,
                'recording'
            ))
            
            conn.commit()
            conn.close()
            print(f"💾 Recording session saved: {session_id}")
            
        except Exception as e:
            print(f"❌ Error saving recording session: {e}")
    
    def _recording_loop(self, session_id: str, audio_file: str):
        """Main recording loop"""
        print(f"🔄 Starting recording loop for session: {session_id}")
        
        try:
            # Open WAV file for writing
            wav_file = wave.open(audio_file, 'wb')
            wav_file.setnchannels(self.channels)
            if self.audio:
                wav_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wav_file.setframerate(self.sample_rate)
            
            chunk_count = 0
            start_time = time.time()
            
            while self.is_recording and self.stream:
                try:
                    # Read audio data
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    wav_file.writeframes(data)
                    
                    chunk_count += 1
                    
                    # Process audio every 10 seconds for transcription
                    if chunk_count % (self.sample_rate // self.chunk_size * 10) == 0:
                        elapsed = time.time() - start_time
                        print(f"🎙️ Recording... {elapsed:.1f}s elapsed")
                        
                        # Process audio chunk for transcription
                        self._process_audio_chunk(session_id, data, chunk_count)
                    
                except Exception as e:
                    print(f"❌ Recording loop error: {e}")
                    break
            
            # Close WAV file
            wav_file.close()
            
            # Update recording session
            duration = time.time() - start_time
            file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
            self._finalize_recording_session(session_id, duration, file_size)
            
            print(f"✅ Recording completed: {duration:.1f}s, {file_size} bytes")
            
            # Automatically process the recording
            asyncio.create_task(self._auto_process_recording(session_id, audio_file))
            
            # Automatically process the recording
            asyncio.create_task(self._auto_process_recording(session_id, audio_file))
            
        except Exception as e:
            print(f"❌ Recording loop error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
        finally:
            # Cleanup
            self._cleanup_audio_resources()
    
    def _process_audio_chunk(self, session_id: str, audio_data: bytes, chunk_number: int):
        """Process audio chunk for real-time transcription"""
        try:
            # This would integrate with speech-to-text API
            # For now, simulate transcription processing
            
            print(f"🔄 Processing audio chunk {chunk_number} for transcription")
            
            # Simulate STT processing delay
            asyncio.create_task(self._simulate_transcription(session_id, chunk_number))
            
        except Exception as e:
            print(f"❌ Audio chunk processing error: {e}")
    
    async def _simulate_transcription(self, session_id: str, chunk_number: int):
        """Simulate transcription processing (replace with real STT)"""
        try:
            # Simulate processing time
            await asyncio.sleep(1)
            
            # Generate simulated transcript
            transcript_text = f"Personal transcript chunk {chunk_number} from recording session {session_id}"
            
            # Store transcript
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO personal_transcripts 
                (recording_session_id, meeting_id, transcript_chunk, timestamp, chunk_number, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                self.current_meeting_id,
                transcript_text,
                datetime.now().isoformat(),
                chunk_number,
                0.85  # Simulated confidence
            ))
            
            conn.commit()
            conn.close()
            
            print(f"💬 Transcript stored: {transcript_text[:50]}...")
            
        except Exception as e:
            print(f"❌ Transcription processing error: {e}")
    
    def stop_recording(self) -> Dict[str, Any]:
        """Stop personal recording"""
        try:
            if not self.is_recording:
                return {"success": False, "error": "No recording in progress"}
            
            print(f"🛑 Stopping personal recording for meeting: {self.current_meeting_id}")
            
            # Stop recording
            self.is_recording = False
            
            # Wait for thread to complete
            if self.recording_thread:
                self.recording_thread.join(timeout=10)
            
            meeting_id = self.current_meeting_id
            self.current_meeting_id = None
            
            print("✅ Personal recording stopped")
            return {
                "success": True,
                "message": f"Personal recording stopped for meeting {meeting_id}"
            }
            
        except Exception as e:
            print(f"❌ Error stopping recording: {e}")
            return {"success": False, "error": str(e)}
    
    def _cleanup_audio_resources(self):
        """Clean up audio resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            if self.audio:
                self.audio.terminate()
                self.audio = None
            
            print("🧹 Audio resources cleaned up")
            
        except Exception as e:
            print(f"❌ Audio cleanup error: {e}")
    
    def _finalize_recording_session(self, session_id: str, duration: float, file_size: int):
        """Finalize recording session in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE personal_meeting_recordings 
                SET end_time = ?, status = ?, duration_seconds = ?, file_size_bytes = ?
                WHERE recording_session_id = ?
            """, (
                datetime.now().isoformat(),
                'completed',
                int(duration),
                file_size,
                session_id
            ))
            
            conn.commit()
            conn.close()
            print(f"📋 Recording session finalized: {session_id}")
            
        except Exception as e:
            print(f"❌ Error finalizing recording: {e}")
    
    async def _auto_process_recording(self, session_id: str, audio_file: str):
        """Automatically process recording when meeting ends"""
        try:
            print(f"🔄 Auto-processing recording: {session_id}")
            
            # Import STT processing
            from .stt_tool import speech_to_text_from_audio
            
            # Process audio file for full transcript
            print(f"🎧 Converting audio to text: {audio_file}")
            
            # Read the audio file as bytes for processing
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            # Get transcript from STT
            transcript_result = speech_to_text_from_audio(audio_data)
            
            if transcript_result.get('success'):
                full_transcript = transcript_result.get('transcript', '')
                confidence = transcript_result.get('confidence', 0.0)
                
                # Save full transcript to database
                await self._save_full_transcript(session_id, full_transcript, confidence)
                
                # Create summary and action items
                await self._create_meeting_summary(session_id, full_transcript)
                
                print(f"✅ Auto-processing completed for {session_id}")
            else:
                print(f"❌ STT processing failed: {transcript_result.get('error')}")
                
        except Exception as e:
            print(f"❌ Auto-processing error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
    
    async def _save_full_transcript(self, session_id: str, transcript: str, confidence: float):
        """Save full transcript to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Update recording session with transcript path
            transcript_file = f"transcripts/{session_id}.txt"
            os.makedirs("transcripts", exist_ok=True)
            
            # Save transcript to file
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            # Update database
            cur.execute("""
                UPDATE personal_meeting_recordings 
                SET transcript_file_path = ?
                WHERE recording_session_id = ?
            """, (transcript_file, session_id))
            
            # Add full transcript entry
            cur.execute("""
                INSERT INTO personal_transcripts 
                (recording_session_id, meeting_id, transcript_chunk, timestamp, chunk_number, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                self.current_meeting_id or 'unknown',
                transcript,
                datetime.now().isoformat(),
                -1,  # -1 indicates full transcript
                confidence
            ))
            
            conn.commit()
            conn.close()
            
            print(f"💾 Full transcript saved: {transcript_file}")
            
        except Exception as e:
            print(f"❌ Error saving transcript: {e}")
    
    async def _create_meeting_summary(self, session_id: str, transcript: str):
        """Create meeting summary and extract action items"""
        try:
            print(f"📝 Creating meeting summary for {session_id}")
            
            # Import processing tools
            from .action_task_tool import extract_action_tasks
            from .todo_tool import extract_and_process_todos
            
            # Extract action tasks from transcript
            action_result = await extract_action_tasks(transcript)
            if action_result.get('success'):
                actions = action_result.get('action_tasks', [])
                print(f"✅ Extracted {len(actions)} action items")
            
            # Extract todos from transcript
            todo_result = await extract_and_process_todos(transcript)
            if todo_result.get('success'):
                todos = todo_result.get('todos', [])
                print(f"✅ Extracted {len(todos)} todo items")
            
            # Store summary in database
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Create meeting_summaries table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meeting_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_session_id TEXT NOT NULL,
                    meeting_id TEXT,
                    summary_text TEXT,
                    action_items TEXT,
                    todo_items TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recording_session_id) REFERENCES personal_meeting_recordings (recording_session_id)
                )
            """)
            
            # Insert summary
            cur.execute("""
                INSERT INTO meeting_summaries 
                (recording_session_id, meeting_id, summary_text, action_items, todo_items)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                self.current_meeting_id or 'unknown',
                f"Meeting recording processed automatically. Duration: {len(transcript)} characters.",
                str(action_result.get('action_tasks', [])),
                str(todo_result.get('todos', []))
            ))
            
            conn.commit()
            conn.close()
            
            print(f"📋 Meeting summary created for {session_id}")
            
        except Exception as e:
            print(f"❌ Error creating summary: {e}")
    
    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status"""
        return {
            "is_recording": self.is_recording,
            "current_meeting_id": self.current_meeting_id,
            "recording_active": self.is_recording,
            "audio_device_ready": self.audio is not None
        }

# Global recorder instance
personal_recorder = PersonalMeetingRecorder()

def start_personal_recording(meeting_id: str, meeting_url: Optional[str] = None) -> Dict[str, Any]:
    """Start personal meeting recording"""
    return personal_recorder.start_recording(meeting_id, meeting_url)

def stop_personal_recording() -> Dict[str, Any]:
    """Stop personal meeting recording"""
    return personal_recorder.stop_recording()

def get_personal_recording_status() -> Dict[str, Any]:
    """Get personal recording status"""
    return personal_recorder.get_recording_status()