# Voice Assistant Server - Production Version

import sqlite3
import aiohttp
import asyncio
import os
import wave
import time
import sys
import signal
from fastmcp import FastMCP
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from dotenv import load_dotenv

# Load environment variables FIRST before importing tools
load_dotenv()

# Import tools with proper path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tools.stt_tool import speech_to_text_from_audio
from tools.notify_tool import notifier, send_periodic_email
from tools.action_task_tool import extract_action_tasks
from tools.calendar_tool import handle_calendar_request, contains_calendar_intent
from tools.mobile_calendar_auth import get_mobile_auth_url, handle_mobile_auth_callback, get_calendar_service_mobile, is_mobile_authenticated
from tools.todo_tool import extract_and_process_todos, todo_manager
from tools.zoom_oauth_tool import get_zoom_auth_url, handle_zoom_oauth_callback, get_zoom_access_token, is_zoom_authenticated
from tools.zoom_webhook_tool import zoom_webhook_handler
from tools.zoom_meeting_tool import detect_zoom_meetings, parse_zoom_meeting_url, process_text_for_zoom_meetings, get_recent_zoom_meetings

# Initialize FastMCP server
server = FastMCP("voice_agent_server")

# Create FastAPI app with large file support for 2+ hour audio
app = FastAPI(
    title="Voice Assistant Server",
    description="Supports long audio processing (2+ hours)",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware for OWASP compliance
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Required OWASP security headers for Zoom integration
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' *.zoom.us virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net; img-src 'self' data: *.zoom.us; connect-src 'self' *.zoom.us virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Permissions-Policy"] = "microphone=*, camera=*, geolocation=()"
    
    return response

# Initialize database
conn = sqlite3.connect("database.db")
cur = conn.cursor()
cur.execute(
    """CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
)
cur.execute(
    """CREATE TABLE IF NOT EXISTS voice_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
)
conn.commit()
conn.close()


async def process_voice(audio_file_path: str) -> dict:
    """
    Process audio with simple speech-to-text and extract structured information
    
    This function:
    1. Transcribes audio using simple STT (no speaker identification)
    2. Extracts todos and action tasks from transcript
    3. Creates calendar events if dates/times mentioned
    4. Returns comprehensive analysis
    
    Args:
        audio_file_path (str): Path to audio file
        
    Returns:
        dict: Complete analysis with transcript information
    """
    try:
        print(f"🎤 Processing audio with simple STT: {audio_file_path}")
        
        # Check if audio file exists
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
        file_size = os.path.getsize(audio_file_path)
        print(f"Audio file size: {file_size:,} bytes")
        
        # Step 1: Simple STT transcription
        with open(audio_file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            
        transcript = await speech_to_text_from_audio(audio_data)
        print(f"📝 Transcript: {transcript}")
        
        # Step 2: Extract structured information from transcript
        if not transcript or len(transcript.strip()) < 10:
            print("⚠️ Warning: Transcript too short or empty")
            return {
                'transcript': transcript or 'No speech detected',
                'speakers': ['Speaker 1'],
                'summary': 'No meaningful content to process',
                'todos': [],
                'action_tasks': [],
                'calendar_result': {'events': []},
                'meeting_summary': 'Audio processing completed but no meaningful speech detected.'
            }
        
        # Step 3: Process with LLM tools for structured extraction
        llm_results = {
            'summary': 'Audio processed with speech-to-text',
            'todos': [],
            'action_tasks': [],
            'calendar_result': {'events': []}
        }
        
        try:
            from tools.llm_tool import call_openai_llm
            summary_prompt = "Please provide a concise summary of this conversation:"
            summary = await call_openai_llm(transcript, summary_prompt)
            llm_results['summary'] = summary
            print(f"✅ Summary generated: {len(summary)} characters")
        except Exception as e:
            print(f"⚠️ Summary generation failed: {e}")
            llm_results['summary'] = "Summary generation failed."
        
        try:
            from tools.todo_tool import extract_and_process_todos
            todo_result = await extract_and_process_todos(transcript)
            todos = todo_result.get('todos', [])
            llm_results['todos'] = todos
            print(f"✅ Todos extracted: {len(todos)} items")
        except Exception as e:
            print(f"⚠️ Todo extraction failed: {e}")
            llm_results['todos'] = []
        
        try:
            from tools.action_task_tool import extract_action_tasks
            action_result = await extract_action_tasks(transcript)
            action_tasks = action_result.get('action_tasks', [])
            llm_results['action_tasks'] = action_tasks
            print(f"✅ Action tasks extracted: {len(action_tasks)} items")
        except Exception as e:
            print(f"⚠️ Action task extraction failed: {e}")
            llm_results['action_tasks'] = []
        
        try:
            from tools.calendar_tool import handle_calendar_request
            calendar_result = await handle_calendar_request(transcript)
            llm_results['calendar_result'] = calendar_result
            events_count = len(calendar_result.get('events', []))
            print(f"✅ Calendar events extracted: {events_count} events")
        except Exception as e:
            print(f"⚠️ Calendar extraction failed: {e}")
            llm_results['calendar_result'] = {'events': []}
        
        # Step 4: Store to database
        try:
            import sqlite3
            conn = sqlite3.connect("database.db")
            cur = conn.cursor()
            cur.execute("INSERT INTO voice_inputs (text) VALUES (?)", (transcript,))
            conn.commit()
            conn.close()
            print("✅ Transcript stored to database")
        except Exception as e:
            print(f"⚠️ Database storage failed: {e}")
        
        # Step 5: Generate meeting summary
        meeting_summary = f"Voice Processing Summary:\\n\\nTranscript: {transcript[:200]}...\\n\\nKey Points: {llm_results.get('summary', 'No summary')}\\n\\nTodos: {len(llm_results.get('todos', []))} items\\nAction Tasks: {len(llm_results.get('action_tasks', []))} items\\nCalendar Events: {len(llm_results.get('calendar_result', {}).get('events', []))} events"
        
        # Return comprehensive results
        return {
            'transcript': transcript,
            'speakers': ['Speaker 1'],  # Single speaker for simple STT
            'summary': llm_results.get("summary", "Audio processed with speech-to-text"),
            'todos': llm_results.get("todos", []),
            'action_tasks': llm_results.get("action_tasks", []),
            'calendar_result': llm_results.get("calendar_result", {'events': []}),
            'meeting_summary': meeting_summary
        }
        
    except Exception as e:
        print(f"❌ Error in voice processing: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'transcript': 'Processing failed',
            'speakers': ['Error'],
            'summary': f'Error: {str(e)}',
            'todos': [],
            'action_tasks': [],
            'calendar_result': {'events': []},
            'meeting_summary': f'Voice processing failed: {str(e)}'
        }


async def speech_to_text_tool(audio_data: bytes) -> str:
    """Convert audio data to text using Azure Speech Service"""
    try:
        from server.tools.stt_tool import speech_to_text_from_audio
    except ImportError:
        from tools.stt_tool import speech_to_text_from_audio
    return await speech_to_text_from_audio(audio_data)


@app.post("/")
async def handle_jsonrpc(request: Request):
    data = await request.json()
    if data.get("method") == "tools/call":
        name = data["params"]["name"]
        args = data["params"]["arguments"]
        try:
            if name == "process_voice":
                result = await process_voice(args["audio_file_path"])
            elif name == "speech_to_text_tool":
                result = await speech_to_text_tool(args["audio_data"])
            elif name == "todo_manager":
                result = await todo_manager(args["item"])
            elif name == "notifier":
                result = await notifier(
                    args["to"], args.get("subject"), args.get("body")
                )
            elif name == "extract_action_tasks":
                result = await extract_action_tasks(args["text"])
            elif name == "extract_and_process_todos":
                result = await extract_and_process_todos(args["text"])
            elif name == "handle_calendar_request":
                result = await handle_calendar_request(args["text"])
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                    "id": data["id"],
                }
            return {"jsonrpc": "2.0", "result": result, "id": data["id"]}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": data["id"],
            }
    return {
        "jsonrpc": "2.0",
        "error": {"code": -32600, "message": "Invalid Request"},
        "id": data["id"],
    }


# ============================================================================
# MOBILE FRONTEND ENDPOINTS - CORE VOICE PROCESSING
# ============================================================================

@app.get("/api/test-stt")
async def test_stt_endpoint():
    """
    🧪 Test endpoint to check STT system status
    """
    try:
        from server.tools.stt_tool import GPT4oHTTPSTT
        
        tool = GPT4oHTTPSTT()
        
        return JSONResponse({
            "success": True,
            "message": "STT system ready",
            "deployment": tool.deployment,
            "api_version": tool.api_version
        })
    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)}, 
            status_code=500
        )


@app.post("/api/process-audio")
async def process_audio_endpoint(audio_file: UploadFile = File(...)):
    """
    🎤 SIMPLE STT AUDIO PROCESSING ENDPOINT
    
    Process audio with simple speech-to-text and extract structured information
    """
    temp_file_path = None
    
    try:
        print(f"🎵 Simple STT audio processing request received")
        print(f"📁 File: {audio_file.filename}, Content Type: {audio_file.content_type}")
        
        # Read audio data from uploaded file
        audio_data = await audio_file.read()
        print(f"📊 Received {len(audio_data)} bytes of audio data")
        
        if not audio_data or len(audio_data) < 44:
            raise HTTPException(
                status_code=400, 
                detail=f"Audio file too small ({len(audio_data)} bytes). Please record longer audio."
            )
        
        # Import STT function
        try:
            from server.tools.stt_tool import speech_to_text_from_audio
            print("✅ Imported STT from server.tools.stt_tool")
        except ImportError:
            from tools.stt_tool import speech_to_text_from_audio
            print("✅ Imported STT from tools.stt_tool")
        
        # Process with STT
        print("🎤 Starting STT processing...")
        
        # Try STT with retry logic for temporary Azure errors
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                transcription = await speech_to_text_from_audio(audio_data)
                print(f"✅ STT processing successful: {transcription[:100]}...")
                break
            except Exception as stt_error:
                if attempt < max_retries and "500" in str(stt_error):
                    print(f"⚠️ Azure 500 error, attempt {attempt + 1}/{max_retries + 1}, retrying...")
                    await asyncio.sleep(1)  # Wait 1 second before retry
                    continue
                else:
                    print(f"❌ STT failed after {attempt + 1} attempts: {stt_error}")
                    transcription = str(stt_error)
                    break
        
        # Check if transcription is valid
        if not transcription or "error" in transcription.lower():
            # Check if it's a temporary Azure error
            if "500" in transcription:
                # Return a more user-friendly response for temporary Azure errors
                result = {
                    "success": True,
                    "transcription": "Audio uploaded successfully, but Azure STT service is temporarily unavailable. Please try again in a moment.",
                    "summary": "Audio processing completed - temporary Azure service issue",
                    "todos": [],
                    "action_tasks": [],
                    "calendar_result": {"success": False, "message": "STT service temporarily unavailable"},
                    "timestamp": time.time(),
                    "processing_info": {
                        "audio_size_bytes": len(audio_data),
                        "transcription_length": 0,
                        "processing_method": "azure_openai_stt_temp_error",
                        "azure_status": "temporary_error_500"
                    }
                }
                print(f"🎯 Returning user-friendly response for Azure 500 error")
                return result
            else:
                raise HTTPException(status_code=400, detail="Speech recognition failed")
        
        # Normal successful transcription
        print(f"🎯 Successfully transcribed: {len(transcription)} characters")
        
        # Step 2: Extract todos, summary, and action tasks from transcription
        print("🧠 Processing transcription for todos, summary, and actions...")
        
        # Extract todos
        todos = []
        try:
            print("📝 Extracting todos...")
            from server.tools.todo_tool import extract_and_process_todos
            todo_result = await extract_and_process_todos(transcription)
            todos = todo_result.get("todos", [])
            summary_from_todos = todo_result.get("summary", "")
            print(f"✅ Extracted {len(todos)} todos")
        except Exception as e:
            print(f"⚠️ Todo extraction failed: {e}")
            summary_from_todos = ""
        
        # Extract action tasks
        action_tasks = []
        try:
            print("⚡ Extracting action tasks...")
            from server.tools.action_task_tool import extract_action_tasks
            action_result = await extract_action_tasks(transcription)
            action_tasks = action_result.get("action_tasks", [])
            print(f"✅ Extracted {len(action_tasks)} action tasks")
        except Exception as e:
            print(f"⚠️ Action task extraction failed: {e}")
        
        # Check for calendar intents
        calendar_result = {"success": False, "message": "No calendar processing"}
        try:
            print("📅 Checking for calendar intents...")
            from server.tools.calendar_tool import contains_calendar_intent, handle_calendar_request
            if contains_calendar_intent(transcription):
                print("🗓️ Calendar intent detected, processing...")
                calendar_result = await handle_calendar_request(transcription)
                print(f"📅 Calendar processing: {calendar_result.get('success', False)}")
        except Exception as e:
            print(f"⚠️ Calendar processing failed: {e}")
        
        # Generate comprehensive summary
        if summary_from_todos:
            summary = summary_from_todos
        else:
            summary = f"Processed voice input ({len(transcription)} characters). Found {len(todos)} todos and {len(action_tasks)} action items."
        
        # Return result
        result = {
            "success": True,
            "transcription": transcription,
            "summary": summary,
            "todos": todos,
            "action_tasks": action_tasks,
            "calendar_result": calendar_result,
            "timestamp": time.time(),
            "processing_info": {
                "audio_size_bytes": len(audio_data),
                "transcription_length": len(transcription),
                "processing_method": "complete_pipeline",
                "todos_count": len(todos),
                "action_tasks_count": len(action_tasks),
                "has_calendar_events": calendar_result.get("success", False)
            }
        }
        
        print(f"🎯 Audio processing completed successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in audio processing: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# Chunked audio storage for large files (2+ hour recordings)
        print(f"   Filename: '{audio_file.filename}'")
        print(f"   Content-Type: '{audio_file.content_type}'")
        print(f"   File size: {audio_file.size if hasattr(audio_file, 'size') else 'unknown'}")
        
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {audio_file.content_type}. Expected audio file."
            )
        
        # Save uploaded file with correct extension based on content type
        import tempfile
        import os
        
        # Map content types to extensions for proper file handling
        content_type_map = {
            'audio/webm': '.webm',
            'audio/wav': '.wav', 
            'audio/mp3': '.mp3',
            'audio/mpeg': '.mp3',
            'audio/mp4': '.m4a',
            'audio/m4a': '.m4a',
            'audio/flac': '.flac'
        }
        
        # Get proper extension based on content type
        detected_extension = content_type_map.get(audio_file.content_type, '.webm')
        print(f"🔍 EXTENSION MAPPING: {audio_file.content_type} -> {detected_extension}")
        
        # Save with correct extension (no conversion needed)
        with tempfile.NamedTemporaryFile(suffix=detected_extension, delete=False) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"💾 Saved as: {temp_file_path}")
        print(f"🔍 File size on disk: {os.path.getsize(temp_file_path)} bytes")
        
        # Validate the saved file
        file_size = os.path.getsize(temp_file_path)
        print(f"🔍 DEBUG: Saved file size: {file_size} bytes")
        
        if file_size < 1000:
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too small ({file_size} bytes). Please record longer audio with clear speech."
            )
        
        # Process with diarization with timeout and fallback
        try:
            print("🎤 Starting STT processing...")
            result = await asyncio.wait_for(
                process_voice(temp_file_path), 
                timeout=120.0  # 2 minute timeout
            )
            print(f"✅ STT processing successful")
            
        except asyncio.TimeoutError:
            print("⚠️ STT timeout - using fallback processing...")
            # Fallback to basic transcription
            try:
                from tools.stt_tool import speech_to_text_from_audio
            except ImportError:
                from server.tools.stt_tool import speech_to_text_from_audio
            
            with open(temp_file_path, 'rb') as audio_data:
                content = audio_data.read()
            
            # Use fallback transcription
            transcription = await speech_to_text_from_audio(content)
            
            # Format as basic response structure
            result = {
                "transcript": transcription,
                "speakers": ["Speaker 1"],
                "summary": "Fallback transcription completed",
                "todos": result.get("todos", []),
                "action_tasks": result.get("action_tasks", []),
                "calendar_result": result.get("calendar_result"),
                "meeting_summary": "Processed with fallback (no speaker identification)",
                "processing_info": {
                    "total_speakers": 0,
                    "total_utterances": 0,
                    "total_todos": len(result.get("todos", [])),
                    "total_action_tasks": len(result.get("action_tasks", [])),
                    "has_calendar_event": bool(result.get("calendar_result", {}).get("success"))
                }
            }
            
        except Exception as e:
            print(f"❌ Diarization processing error: {e}")
            # Fallback to regular processing on any error
            try:
                from tools.stt_tool import speech_to_text_from_audio
            except ImportError:
                from server.tools.stt_tool import speech_to_text_from_audio
            
            with open(temp_file_path, 'rb') as audio_data:
                content = audio_data.read()
            
            transcription = await speech_to_text_from_audio(content)
            fallback_result = await process_voice(transcription)
            
            result = {
                "diarized_transcript": {
                    "full_text": transcription,
                    "speakers": [],
                    "utterances": [],
                    "speaker_statistics": {}
                },
                "summary": fallback_result.get("summary", ""),
                "todos": fallback_result.get("todos", []),
                "action_tasks": fallback_result.get("action_tasks", []),
                "calendar_result": fallback_result.get("calendar_result"),
                "meeting_summary": f"Processed with fallback due to error: {str(e)[:100]}",
                "processing_info": {
                    "total_speakers": 0,
                    "total_utterances": 0,
                    "total_todos": len(fallback_result.get("todos", [])),
                    "total_action_tasks": len(fallback_result.get("action_tasks", [])),
                    "has_calendar_event": bool(fallback_result.get("calendar_result", {}).get("success"))
                }
            }
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

# Chunked audio storage for large files (2+ hour recordings)
chunk_storage = {}

@app.post("/api/process-audio-chunk")
async def process_audio_chunk_endpoint(
    audio_file: UploadFile = File(...),
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    is_final_chunk: bool = Form(False)
):
    """
    🎯 CHUNKED AUDIO PROCESSING - For 2+ Hour Recordings
    
    Handles large audio files by processing them in chunks and combining results.
    This prevents memory issues and timeout problems with very long recordings.
    """
    try:
        # Validate required parameters
        if not session_id or session_id.strip() == "":
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if chunk_index < 0 or total_chunks <= 0:
            raise HTTPException(status_code=400, detail="Invalid chunk parameters")
        
        print(f"📦 Processing chunk {chunk_index + 1}/{total_chunks} for session {session_id}")
        
        # Read chunk data
        chunk_data = await audio_file.read()
        if not chunk_data:
            raise HTTPException(status_code=400, detail="No audio data in chunk")
        
        # Initialize storage for this session if needed
        if session_id not in chunk_storage:
            chunk_storage[session_id] = {
                "chunks": {},
                "transcriptions": [],
                "total_chunks": total_chunks,
                "received_chunks": 0
            }
        
        # Store chunk data
        chunk_storage[session_id]["chunks"][chunk_index] = chunk_data
        chunk_storage[session_id]["received_chunks"] += 1
        
        # Process this chunk for transcription
        try:
            chunk_text = await asyncio.wait_for(
                speech_to_text_from_audio(chunk_data), 
                timeout=1800  # 30 minutes per chunk
            )
            
            if chunk_text and chunk_text.strip():
                chunk_storage[session_id]["transcriptions"].append(chunk_text)
                print(f"✅ Chunk {chunk_index + 1} transcribed successfully")
            
        except asyncio.TimeoutError:
            print(f"⏰ Chunk {chunk_index + 1} transcription timed out")
            chunk_storage[session_id]["transcriptions"].append("")
        
        # If this is the final chunk, process complete audio
        if is_final_chunk and chunk_storage[session_id]["received_chunks"] == total_chunks:
            print(f"🎯 Processing final chunk - combining all transcriptions")
            
            # Combine all transcriptions
            full_transcription = " ".join(
                [t for t in chunk_storage[session_id]["transcriptions"] if t.strip()]
            )
            
            if not full_transcription.strip():
                # Clean up storage
                del chunk_storage[session_id]
                raise HTTPException(status_code=400, detail="No speech detected in any chunks")
            
            print(f"📝 Complete transcription: {len(full_transcription)} characters")
            
            # Process with LLM and tools
            try:
                result = await asyncio.wait_for(
                    process_voice(full_transcription), 
                    timeout=1800  # 30 minutes for large text processing
                )
                
                # Clean up storage
                del chunk_storage[session_id]
                
                return {
                    "success": True,
                    "transcription": full_transcription,
                    "summary": result["summary"],
                    "todos": result["todos"],
                    "action_tasks": result["action_tasks"],
                    "calendar_result": result.get("calendar_result"),
                    "timestamp": time.time(),
                    "chunks_processed": total_chunks
                }
                
            except asyncio.TimeoutError:
                # Clean up storage
                del chunk_storage[session_id]
                raise HTTPException(status_code=408, detail="Large text processing timed out")
        else:
            # Return progress for intermediate chunks
            return {
                "success": True,
                "chunk_processed": chunk_index + 1,
                "total_chunks": total_chunks,
                "transcription": chunk_storage[session_id]["transcriptions"][-1] if chunk_storage[session_id]["transcriptions"] else "",
                "progress": (chunk_index + 1) / total_chunks * 100
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing chunk: {e}")
        # Clean up storage on error
        if session_id in chunk_storage:
            del chunk_storage[session_id]
        raise HTTPException(status_code=500, detail=f"Chunk processing failed: {str(e)}")


@app.get("/api/health")
async def health_check():
    """
    🏥 HEALTH CHECK ENDPOINT
    
    FLUTTER INTEGRATION:
    ===================
    
    HTTP Method: GET
    URL: {base_url}/api/health
    
    Response:
    --------
    {
        "status": "healthy",
        "timestamp": 1699999999.999
    }
    
    Usage:
    -----
    - Call this before making other API requests
    - Use for connection testing and server status monitoring
    - Implement retry logic if this fails
    
    Flutter Example:
    ---------------
    ```dart
    try {
        var response = await http.get(Uri.parse('$baseUrl/api/health'));
        if (response.statusCode == 200) {
            var data = json.decode(response.body);
            print('Server status: ${data['status']}');
        }
    } catch (e) {
        print('Server not reachable: $e');
    }
    ```
    """
    return {"status": "healthy", "timestamp": time.time()}


# Original JSONRPC endpoint for compatibility

@app.get("/api/auth/google/url")
async def get_google_auth_url():
    """
    🔐 GOOGLE CALENDAR AUTHENTICATION - Get OAuth URL
    
    FLUTTER INTEGRATION:
    ===================
    
    HTTP Method: GET
    URL: {base_url}/api/auth/google/url
    
    Response (Success):
    -----------------
    {
        "success": true,
        "authorization_url": "https://accounts.google.com/o/oauth2/auth?...",
        "state": "random_state_string"
    }
    
    Response (Error):
    ---------------
    {
        "success": false,
        "error": "Error message"
    }
    
    Flutter Implementation Steps:
    ===========================
    
    1. Call this endpoint to get authorization URL
    2. Open URL in webview or external browser:
       ```dart
       import 'package:url_launcher/url_launcher.dart';
       
       await launchUrl(Uri.parse(authUrl), mode: LaunchMode.externalApplication);
       ```
    3. User logs in with Google and grants calendar permissions
    4. Google redirects to backend /oauth/callback (handled automatically)
    5. Check authentication status with /api/auth/status
    
    WebView Option:
    --------------
    ```dart
    WebView(
        initialUrl: authUrl,
        onPageFinished: (url) {
            if (url.contains('success') || url.contains('callback')) {
                // Close webview and check auth status
                checkAuthStatus();
            }
        }
    )
    ```
    
    Note: Calendar features require this authentication step!
    """
    try:
        auth_data = get_mobile_auth_url()
        if auth_data:
            return JSONResponse({
                "success": True,
                "authorization_url": auth_data["authorization_url"],
                "state": auth_data["state"]
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Failed to generate authorization URL"
            }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/oauth/callback")
async def oauth_callback(request: Request):
    """
    🔄 OAUTH CALLBACK ENDPOINT (Auto-handled by Google)
    
    FLUTTER INTEGRATION:
    ===================
    
    ⚠️  IMPORTANT: You don't call this endpoint directly!
    
    This endpoint is automatically called by Google after user authorization.
    
    How it works:
    ============
    1. User clicks auth URL from /api/auth/google/url
    2. User logs in with Google
    3. User grants calendar permissions
    4. Google automatically redirects to: {base_url}/oauth/callback?code=...&state=...
    5. This endpoint processes the callback
    6. Authentication is complete
    
    Query Parameters (provided by Google):
    ====================================
    - code: Authorization code (used to get access tokens)
    - state: Security state parameter
    - error: Error message (if user denied access)
    
    Response:
    ========
    {
        "success": true/false,
        "message": "Authentication successful" or error message
    }
    
    Flutter Monitoring:
    ==================
    
    If using WebView, you can monitor for this URL:
    ```dart
    WebView(
        onPageFinished: (url) {
            if (url.contains('/oauth/callback')) {
                // Authentication flow completed
                // Close WebView and check auth status
                Navigator.pop(context);
                checkAuthenticationStatus();
            }
        }
    )
    ```
    
    Or if using external browser:
    ```dart
    // After launching auth URL, periodically check auth status
    Timer.periodic(Duration(seconds: 2), (timer) async {
        bool isAuth = await checkAuthStatus();
        if (isAuth) {
            timer.cancel();
            // Update UI to show calendar is connected
            setState(() { calendarConnected = true; });
        }
    });
    ```
    """
    try:
        code = request.query_params.get('code')
        state = request.query_params.get('state')
        error = request.query_params.get('error')
        
        if error:
            return JSONResponse({
                "success": False,
                "error": f"OAuth error: {error}"
            }, status_code=400)
        
        if not code or not state:
            return JSONResponse({
                "success": False,
                "error": "Missing authorization code or state"
            }, status_code=400)
        
        # Handle the OAuth callback
        success = handle_mobile_auth_callback(code, state)
        
        if success:
            return JSONResponse({
                "success": True,
                "message": "Authentication successful"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Failed to process authentication"
            }, status_code=500)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/auth/status")
async def check_auth_status():
    """
    📊 CHECK GOOGLE CALENDAR AUTHENTICATION STATUS
    
    FLUTTER INTEGRATION:
    ===================
    
    HTTP Method: GET
    URL: {base_url}/api/auth/status
    
    Response:
    --------
    {
        "authenticated": true/false,
        "message": "Description of current state"
    }
    
    Flutter Example:
    ---------------
    ```dart
    bool isGoogleConnected = await checkAuthStatus();
    if (!isGoogleConnected) {
        showGoogleAuthDialog();
    }
    ```
    """
    try:
        authenticated = is_mobile_authenticated()
        return JSONResponse({
            "authenticated": authenticated,
            "message": "Ready to create calendar events" if authenticated else "Authentication required"
        })
    except Exception as e:
        return JSONResponse({
            "authenticated": False,
            "error": str(e)
        }, status_code=500)

# ================================
# ZOOM INTEGRATION API ENDPOINTS
# ================================

@app.get("/api/zoom/auth/url")
async def get_zoom_auth_url_api():
    """
    🔗 ZOOM OAUTH AUTHENTICATION - Get Authorization URL
    
    FLUTTER INTEGRATION:
    ===================
    
    HTTP Method: GET
    URL: {base_url}/api/zoom/auth/url
    
    Response (Success):
    -----------------
    {
        "success": true,
        "authorization_url": "https://zoom.us/oauth/authorize?...",
        "state": "random_state_string"
    }
    
    Response (Error):
    ---------------
    {
        "success": false,
        "error": "Error message"
    }
    
    Flutter Implementation:
    =====================
    
    ```dart
    Future<void> connectZoom() async {
        try {
            var response = await http.get(Uri.parse('$baseUrl/api/zoom/auth/url'));
            var data = json.decode(response.body);
            
            if (data['success']) {
                // Open Zoom auth in browser
                await launchUrl(Uri.parse(data['authorization_url']));
                
                // Start polling for auth completion
                _pollZoomAuthStatus();
            }
        } catch (e) {
            showError('Failed to start Zoom authentication: $e');
        }
    }
    ```
    """
    try:
        auth_data = get_zoom_auth_url()
        return JSONResponse(auth_data)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# Removed duplicate OAuth callback route - using /api/zoom/auth/callback instead

@app.get("/api/zoom/auth/status")
async def check_zoom_auth_status():
    """
    📊 CHECK ZOOM AUTHENTICATION STATUS
    
    FLUTTER INTEGRATION:
    ===================
    
    HTTP Method: GET
    URL: {base_url}/api/zoom/auth/status
    
    Response:
    --------
    {
        "authenticated": true/false,
        "message": "Description of current state",
        "expires_at": "2025-01-15T10:30:00",  // Optional
        "scope": "meeting:read meeting:write"  // Optional
    }
    
    Flutter Example:
    ---------------
    ```dart
    Future<bool> isZoomConnected() async {
        try {
            var response = await http.get(Uri.parse('$baseUrl/api/zoom/auth/status'));
            var data = json.decode(response.body);
            return data['authenticated'] ?? false;
        } catch (e) {
            return false;
        }
    }
    ```
    """
    try:
        auth_status = is_zoom_authenticated()
        return JSONResponse(auth_status)
    except Exception as e:
        return JSONResponse({
            "authenticated": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/zoom/auth/callback")
async def zoom_oauth_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    """
    🔄 ZOOM OAUTH CALLBACK HANDLER
    
    This endpoint handles the OAuth callback from Zoom after user authorization.
    Zoom redirects here with the authorization code.
    """
    try:
        if error:
            return HTMLResponse(f"""
            <html><body>
                <h1>❌ Zoom Authorization Failed</h1>
                <p>Error: {error}</p>
                <p><a href="javascript:window.close()">Close this window</a></p>
            </body></html>
            """)
        
        if not code:
            return HTMLResponse(f"""
            <html><body>
                <h1>❌ Missing Authorization Code</h1>
                <p>No authorization code received from Zoom.</p>
                <p><a href="javascript:window.close()">Close this window</a></p>
            </body></html>
            """)
        
        # Exchange code for tokens using zoom_oauth_tool
        from tools.zoom_oauth_tool import zoom_oauth_manager
        result = await zoom_oauth_manager.exchange_code_for_tokens(code, state)
        
        if result.get("success"):
            return HTMLResponse(f"""
            <html><body>
                <h1>✅ Zoom Connected Successfully!</h1>
                <p>Your virtual assistant is now connected to Zoom.</p>
                <p>You can now close this window and join any Zoom meeting for automatic processing.</p>
                <script>
                    setTimeout(() => window.close(), 3000);
                </script>
            </body></html>
            """)
        else:
            error_msg = result.get("error", "Token exchange failed")
            return HTMLResponse(f"""
            <html><body>
                <h1>❌ Token Exchange Failed</h1>
                <p>Error: {error_msg}</p>
                <p><a href="javascript:window.close()">Close this window</a></p>
            </body></html>
            """)
    
    except Exception as e:
        return HTMLResponse(f"""
        <html><body>
            <h1>❌ Callback Error</h1>
            <p>Error: {str(e)}</p>
            <p><a href="javascript:window.close()">Close this window</a></p>
        </body></html>
        """)

@app.post("/api/zoom/meetings/detect")
async def detect_zoom_meetings_api(request: Request):
    """
    🔍 DETECT ZOOM MEETINGS IN TEXT
    
    FLUTTER INTEGRATION:
    ===================
    
    HTTP Method: POST
    URL: {base_url}/api/zoom/meetings/detect
    Content-Type: application/json
    
    Request Body:
    -----------
    {
        "text": "Join our meeting at https://zoom.us/j/123456789?pwd=abc123"
    }
    
    Response (Success):
    -----------------
    {
        "meetings_detected": true,
        "meetings_count": 1,
        "meetings": [
            {
                "original_url": "https://zoom.us/j/123456789?pwd=abc123",
                "meeting_id": "123456789",
                "passcode": "abc123",
                "domain": "zoom.us",
                "is_valid": true,
                "success": true,
                "topic": "Weekly Standup",
                "start_time": "2025-01-15T10:00:00",
                "join_url": "...",
                "detected_from_url": "..."
            }
        ]
    }
    
    Flutter Implementation:
    =====================
    
    ```dart
    Future<List<ZoomMeeting>> detectMeetings(String text) async {
        try {
            var response = await http.post(
                Uri.parse('$baseUrl/api/zoom/meetings/detect'),
                headers: {'Content-Type': 'application/json'},
                body: json.encode({'text': text})
            );
            
            var data = json.decode(response.body);
            if (data['meetings_detected']) {
                return data['meetings'].map<ZoomMeeting>(
                    (m) => ZoomMeeting.fromJson(m)
                ).toList();
            }
            return [];
        } catch (e) {
            print('Error detecting meetings: $e');
            return [];
        }
    }
    ```
    """
    try:
        body = await request.json()
        text = body.get("text", "")
        
        if not text:
            return JSONResponse({
                "meetings_detected": False,
                "error": "No text provided"
            }, status_code=400)
        
        # Process text for Zoom meetings
        result = await process_text_for_zoom_meetings(text)
        
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse({
            "meetings_detected": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/zoom/meetings/recent")
async def get_recent_meetings_api(request: Request):
    """
    📋 GET RECENT ZOOM MEETINGS
    
    FLUTTER INTEGRATION:
    ===================
    
    HTTP Method: GET
    URL: {base_url}/api/zoom/meetings/recent?limit=10
    
    Query Parameters:
    ---------------
    - limit: Number of meetings to return (optional, default 10)
    
    Response:
    --------
    {
        "success": true,
        "meetings": [
            {
                "meeting_id": "123456789",
                "topic": "Weekly Standup",
                "join_url": "https://zoom.us/j/123456789",
                "start_time": "2025-01-15T10:00:00",
                "status": "upcoming",
                "created_at": "2025-01-10T15:30:00"
            }
        ],
        "count": 5
    }
    
    Flutter Example:
    ---------------
    ```dart
    Future<List<ZoomMeeting>> getRecentMeetings() async {
        try {
            var response = await http.get(
                Uri.parse('$baseUrl/api/zoom/meetings/recent?limit=20')
            );
            var data = json.decode(response.body);
            
            if (data['success']) {
                return data['meetings'].map<ZoomMeeting>(
                    (m) => ZoomMeeting.fromJson(m)
                ).toList();
            }
            return [];
        } catch (e) {
            return [];
        }
    }
    ```
    """
    try:
        limit = int(request.query_params.get('limit', 10))
        meetings = get_recent_zoom_meetings(limit)
        
        return JSONResponse({
            "success": True,
            "meetings": meetings,
            "count": len(meetings)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/webhooks/zoom")
async def zoom_webhook_endpoint(request: Request):
    """
    🪝 ZOOM WEBHOOK HANDLER
    
    This endpoint receives webhook events from Zoom.
    Configure this URL in your Zoom app webhook settings:
    https://your-server.com/webhooks/zoom
    
    Handles Events:
    - meeting.started
    - meeting.ended  
    - meeting.participant_joined
    - meeting.participant_left
    - recording.transcript_completed
    
    This is called by Zoom, not your Flutter app directly.
    """
    try:
        # Process webhook with signature verification
        result = await zoom_webhook_handler.handle_webhook_event(request)
        
        if result.get("success"):
            return JSONResponse({"success": True})
        else:
            return JSONResponse({
                "success": False,
                "error": result.get("error")
            }, status_code=400)
            
    except Exception as e:
        print(f"❌ Webhook processing error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/webhooks/zoom/chat")
async def zoom_chat_bot_endpoint(request: Request):
    """
    💬 ZOOM TEAM CHAT BOT ENDPOINT
    
    This endpoint receives chat bot events from Zoom Team Chat.
    Configure this URL in your Zoom app chat bot settings:
    https://your-server.com/webhooks/zoom/chat
    
    Handles Events:
    - bot_installed
    - bot_notification
    - interactive_message_select
    - slash_command
    
    This is called by Zoom Team Chat, not your Flutter app directly.
    """
    try:
        # Get request data
        body = await request.body()
        headers = dict(request.headers)
        
        print(f"💬 Received chat bot event: {len(body)} bytes")
        
        # Parse the chat event
        try:
            import json
            event_data = json.loads(body.decode())
            event_type = event_data.get('event')
            
            print(f"📝 Chat event type: {event_type}")
            
            # Handle different chat bot events
            if event_type == 'bot_notification':
                # User sent a message to the bot
                message = event_data.get('payload', {}).get('plainToken', '')
                user_id = event_data.get('payload', {}).get('userId', '')
                to_jid = event_data.get('payload', {}).get('toJid', '')
                
                print(f"💬 Message from {user_id}: {message}")
                
                # Process the message with voice assistant
                response_text = await process_chat_message(message)
                
                # Send response back to chat
                await send_chat_response(to_jid, response_text)
                
            elif event_type == 'slash_command':
                # Handle slash commands like /voice, /todo, etc.
                command = event_data.get('payload', {}).get('cmd', '')
                user_id = event_data.get('payload', {}).get('userId', '')
                to_jid = event_data.get('payload', {}).get('toJid', '')
                
                print(f"⚡ Slash command from {user_id}: {command}")
                
                response_text = await handle_slash_command(command)
                await send_chat_response(to_jid, response_text)
                
            elif event_type == 'bot_installed':
                print(f"🎉 Bot installed by user")
                
            return JSONResponse({"success": True})
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in chat event: {e}")
            return JSONResponse({
                "success": False,
                "error": "Invalid JSON"
            }, status_code=400)
            
    except Exception as e:
        print(f"❌ Chat bot processing error: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

async def process_chat_message(message: str) -> str:
    """Process chat message and return response"""
    try:
        # Detect if message contains Zoom meeting URLs
        meetings = detect_zoom_meetings(message)
        if meetings:
            return f"🔗 I found {len(meetings)} Zoom meeting(s) in your message. Would you like me to process them?"
        
        # Simple responses for common commands
        message_lower = message.lower()
        if 'hello' in message_lower or 'hi' in message_lower:
            return "👋 Hello! I'm your voice assistant. Send me voice commands or Zoom meeting links!"
        elif 'help' in message_lower:
            return "💡 I can help you with:\n• Voice processing\n• Zoom meeting detection\n• Calendar events\n• Todo management\nTry sending me a Zoom meeting link!"
        else:
            return f"📝 I received your message: '{message}'. Send me voice commands or Zoom meeting links for processing!"
            
    except Exception as e:
        print(f"❌ Error processing chat message: {e}")
        return "❌ Sorry, I had trouble processing your message."

async def handle_slash_command(command: str) -> str:
    """Handle slash commands from chat"""
    try:
        command_lower = command.lower()
        
        if command_lower.startswith('voice'):
            return "🎤 Voice Assistant is ready! Send me audio files or Zoom meeting links."
        elif command_lower.startswith('todo'):
            return "📝 Todo management available. Send voice commands to create todos."
        elif command_lower.startswith('calendar'):
            return "📅 Calendar integration active. I can create events from voice commands."
        elif command_lower.startswith('zoom'):
            return "🔗 Zoom integration ready! Send meeting links for automatic processing."
        else:
            return f"❓ Unknown command: {command}\nAvailable: /voice, /todo, /calendar, /zoom"
            
    except Exception as e:
        print(f"❌ Error handling slash command: {e}")
        return "❌ Sorry, I couldn't process that command."

async def send_chat_response(to_jid: str, message: str):
    """Send response message back to Zoom chat"""
    try:
        # This would need Zoom chat API implementation
        # For now, just log the response
        print(f"📤 Would send to {to_jid}: {message}")
        # TODO: Implement actual Zoom chat API call
        
    except Exception as e:
        print(f"❌ Error sending chat response: {e}")

@app.post("/api/calendar/create")
async def create_calendar_event_api(request: Request):
    """
    📅 MANUAL CALENDAR EVENT CREATION
    
    FLUTTER INTEGRATION:
    ===================
    
    HTTP Method: POST
    URL: {base_url}/api/calendar/create
    Content-Type: application/json
    
    Request Body:
    -----------
    {
        "task": "Meeting with client",
        "date": "2025-11-15",  // Format: YYYY-MM-DD
        "time": "14:00"        // Format: HH:MM (24-hour), optional, defaults to 09:00
    }
    
    Response (Success):
    -----------------
    {
        "success": true,
        "message": "Calendar event created: 'Meeting with client' scheduled for 2025-11-15 at 14:00",
        "event_data": {
            "task": "Meeting with client",
            "date": "2025-11-15",
            "time": "14:00",
            "event_id": 123
        },
        "google_calendar": {
            "success": true,
            "message": "Event created in Google Calendar: https://calendar.google.com/...",
            "event_id": "abc123..."
        }
    }
    
    Response (Not Authenticated):
    ---------------------------
    {
        "success": false,
        "error": "Not authenticated with Google Calendar"
    }
    Status: 401
    
    Response (Validation Error):
    --------------------------
    {
        "success": false,
        "error": "Task and date are required"
    }
    Status: 400
    
    Flutter Usage:
    =============
    
    ```dart
    Future<Map<String, dynamic>> createCalendarEvent({
        required String task,
        required String date,  // YYYY-MM-DD
        String time = '09:00'  // HH:MM
    }) async {
        var body = json.encode({
            'task': task,
            'date': date,
            'time': time,
        });
        
        var response = await http.post(
            Uri.parse('$baseUrl/api/calendar/create'),
            headers: {'Content-Type': 'application/json'},
            body: body,
        );
        
        var result = json.decode(response.body);
        
        if (response.statusCode == 401) {
            // Need to authenticate
            await authenticateGoogleCalendar();
            return {'success': false, 'error': 'Authentication required'};
        }
        
        return result;
    }
    
    // Usage example:
    var result = await createCalendarEvent(
        task: 'Team meeting',
        date: '2025-11-18',
        time: '15:30'
    );
    
    if (result['success']) {
        showSuccessMessage(result['message']);
        // Optionally open Google Calendar link
        var calendarLink = result['google_calendar']['message'];
    } else {
        showErrorMessage(result['error']);
    }
    ```
    
    Prerequisites:
    =============
    - User must be authenticated with Google Calendar
    - Check authentication with /api/auth/status first
    - If not authenticated, use /api/auth/google/url flow
    """
    try:
        data = await request.json()
        task = data.get('task', '')
        date = data.get('date', '')
        time = data.get('time', '09:00')
        
        if not task or not date:
            return JSONResponse({
                "success": False,
                "error": "Task and date are required"
            }, status_code=400)
        
        # Check if authenticated
        if not is_mobile_authenticated():
            return JSONResponse({
                "success": False,
                "error": "Not authenticated with Google Calendar"
            }, status_code=401)
        
        # Import calendar tool
        try:
            from tools.calendar_tool import CalendarTool
        except ImportError:
            from server.tools.calendar_tool import CalendarTool
        calendar_tool = CalendarTool()
        
        # Create the event
        result = await calendar_tool.create_simple_calendar_event(task, date, time)
        
        return JSONResponse({
            "success": result["success"],
            "message": result["message"],
            "event_data": {
                "task": result.get("task"),
                "date": result.get("date"),
                "time": result.get("time"),
                "event_id": result.get("event_id")
            },
            "google_calendar": result.get("google_calendar", {})
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


if __name__ == "__main__":
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\\n🔄 Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    async def main():
        """
        Main function to run the MCP server with error handling.
        """
        print("🚀 Starting HTTP-based Voice Assistant Server...")

        try:
            # Initialize Zoom database tables
            print("🔧 Initializing Zoom database tables...")
            from tools.zoom_oauth_tool import zoom_oauth_manager
            zoom_oauth_manager._init_database()
            print("✅ Zoom database tables initialized")
            
            # Start ASGI server with error handling
            config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
            asgi_server = uvicorn.Server(config)
            print("✅ HTTP Voice Assistant server started on port 8000")
            
            # Print available endpoints
            print("\n🌐 Available Zoom Integration Endpoints:")
            print("   📋 GET  /api/zoom/auth/status - Check Zoom authentication")
            print("   🔗 GET  /api/zoom/auth/url - Get Zoom OAuth URL") 
            print("   🔄 GET  /api/zoom/auth/callback - Zoom OAuth callback")
            print("   🔍 POST /api/zoom/meetings/detect - Detect meetings in text")
            print("   📊 GET  /api/zoom/meetings/recent - Get recent meetings")
            print("   🪝 POST /webhooks/zoom - Zoom webhook handler")
            print("\n🔑 Environment variables needed for Zoom:")
            print("   ZOOM_CLIENT_ID - Your Zoom app client ID")
            print("   ZOOM_CLIENT_SECRET - Your Zoom app client secret")
            print("   ZOOM_WEBHOOK_SECRET - Your Zoom webhook verification token")

            # Start periodic email task
            email_task = asyncio.create_task(send_periodic_email())

            # Run the ASGI server (this runs the event loop)
            await asgi_server.serve()
        except Exception as main_error:
            print(f"❌ Main server startup error: {main_error}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            raise  # Re-raise to be caught by outer handler

    # Run the main function with comprehensive error handling and auto-restart
    restart_count = 0
    max_restarts = 3
    
    while restart_count < max_restarts:
        try:
            print(f"🚀 Initializing HTTP Voice Assistant server (attempt {restart_count + 1}/{max_restarts})...")
            # Initialize FastMCP tools
            server.tool()(process_voice)
            server.tool()(speech_to_text_tool)
            server.tool()(todo_manager)
            server.tool()(notifier)
            server.tool()(extract_action_tasks)
            server.tool()(extract_and_process_todos)
            server.tool()(handle_calendar_request)
            
            asyncio.run(main())
            # If we get here, server stopped normally
            break
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user (Ctrl+C)")
            print("👋 Goodbye!")
            break
        except Exception as e:
            restart_count += 1
            print(f"💥 Critical server error: {e}")
            import traceback
            print(f"🔍 Full traceback:\n{traceback.format_exc()}")
            
            if restart_count < max_restarts:
                print(f"🔄 Attempting auto-restart ({restart_count}/{max_restarts})...")
                time.sleep(2)  # Wait before restart
            else:
                print(f"❌ Maximum restart attempts reached ({max_restarts})")
                print("🛑 Server shutdown - manual intervention required")
                exit(1)
        finally:
            print("🔒 Server cleanup complete")
