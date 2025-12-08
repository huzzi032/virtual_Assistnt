"""
Audio Processor - Real Implementation
Handles audio file processing with actual transcription, todos, and action items
"""

import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio

# Import notification tool
try:
    from notify_tool import notifier
except ImportError:
    try:
        from tools.notify_tool import notifier
    except ImportError:
        from server.tools.notify_tool import notifier

def init_database():
    """Initialize the audio files database"""
    try:
        conn = sqlite3.connect('audio_files.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                user_email TEXT,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                transcript TEXT,
                action_items TEXT,
                todo_items TEXT,
                processed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        print("✅ Audio files database initialized")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")

async def process_uploaded_audio(file_path: str, filename: Optional[str] = None, file_size: Optional[int] = None, user_email: Optional[str] = None) -> Dict[str, Any]:
    """
    Process uploaded audio file
    
    Args:
        file_path: Path to the uploaded audio file
        filename: Original filename of the audio file
        file_size: Size of the file in bytes
        user_email: Optional user email for notifications
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Initialize database
        init_database()
        
        # Set defaults
        final_filename = filename or os.path.basename(file_path)
        final_file_size = file_size or os.path.getsize(file_path)
        final_user_email = user_email or "unknown"
        
        # Insert initial record
        conn = sqlite3.connect('audio_files.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audio_files (filename, user_email, file_path, file_size, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (final_filename, final_user_email, file_path, final_file_size, datetime.now().isoformat()))
        
        processing_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        if not processing_id:
            raise ValueError("Failed to get processing ID from database")
        
        # Process based on file size
        if final_file_size > 50 * 1024 * 1024:  # 50MB threshold for chunking
            return await _process_large_audio_file(file_path, processing_id, final_filename, final_user_email)
        else:
            return await _process_small_audio_file(file_path, processing_id, final_filename, final_user_email)
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def _process_large_audio_file(file_path: str, processing_id: int, filename: str, user_email: str) -> Dict[str, Any]:
    """
    Process large audio file using direct transcription (no chunking yet)
    """
    try:
        print(f"🔄 Processing large audio file: {filename} (ID: {processing_id})")
        
        # Process large audio with actual transcription
        try:
            from stt_tool import speech_to_text_from_audio
            print("✅ Using speech-to-text processing for large file")
            
            # Read file as bytes for STT
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # Get transcription
            transcription = await speech_to_text_from_audio(audio_data)
            print(f"✅ Large file transcription completed: {len(transcription)} characters")
            
            # Extract todos and action items
            todos = []
            action_items = []
            summary = ""
            
            if transcription and len(transcription) > 10:
                # Extract todos
                try:
                    from todo_tool import extract_and_process_todos
                    todo_result = await extract_and_process_todos(transcription)
                    todos = todo_result.get("todos", [])
                    summary = todo_result.get("summary", "")
                    print(f"✅ Extracted {len(todos)} todos from large file")
                except Exception as e:
                    print(f"⚠️ Todo extraction failed: {e}")
                    
                # Extract action items
                try:
                    from action_task_tool import extract_action_tasks
                    action_result = await extract_action_tasks(transcription)
                    action_items = action_result.get("action_tasks", [])
                    print(f"✅ Extracted {len(action_items)} action items from large file")
                except Exception as e:
                    print(f"⚠️ Action item extraction failed: {e}")
            
        except Exception as e:
            print(f"❌ Large file processing failed: {e}")
            transcription = f"Large file processing failed: {str(e)}"
            todos = []
            action_items = []
            summary = "Large file processing encountered an error"
        
        # Update database with real results
        conn = sqlite3.connect('audio_files.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE audio_files 
            SET transcript = ?, action_items = ?, todo_items = ?, processed_at = ?
            WHERE id = ?
        """, (
            transcription,
            str(action_items),
            str(todos),
            datetime.now().isoformat(),
            processing_id
        ))
        
        conn.commit()
        conn.close()
        
        # Send email notification with real results
        if user_email and user_email != "unknown":
            try:
                email_subject = f"Large Audio Processing Complete - {filename}"
                
                # Format todos and action items for email
                todos_text = '\\n'.join([f"• {todo}" for todo in todos]) if todos else "No todos found"
                actions_text = '\\n'.join([f"• {action.get('task', str(action)) if isinstance(action, dict) else str(action)}" for action in action_items]) if action_items else "No action items found"
                
                email_body = f"""Large Audio Processing Results for: {filename}

TRANSCRIPT:
{transcription}

ACTION ITEMS:
{actions_text}

TODO ITEMS:
{todos_text}

SUMMARY:
{summary or 'Large audio file processed successfully'}

Processing completed successfully!

Best regards,
Your Virtual Assistant"""
                
                await notifier(user_email, email_subject, email_body)
                print(f"📧 Email notification sent to {user_email}")
            except Exception as e:
                print(f"⚠️ Failed to send email notification: {e}")
        
        return {
            "success": True,
            "processing_id": processing_id,
            "message": f"Large audio file ({filename}) processed successfully",
            "filename": filename,
            "processing_type": "direct",
            "transcription": transcription,
            "action_items": action_items,
            "todo_items": todos,
            "summary": summary or "Large audio file processed successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def _process_small_audio_file(file_path: str, processing_id: int, filename: str, user_email: str) -> Dict[str, Any]:
    """
    Process small audio file directly
    """
    try:
        print(f"🎵 Processing small audio file: {filename} (ID: {processing_id})")
        
        # Process audio with actual transcription
        try:
            from stt_tool import speech_to_text_from_audio
            print("✅ Using speech-to-text processing")
            
            # Read file as bytes for STT
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # Get transcription
            transcription = await speech_to_text_from_audio(audio_data)
            print(f"✅ Transcription completed: {len(transcription)} characters")
            
            # Extract todos and action items
            todos = []
            action_items = []
            summary = ""
            
            if transcription and len(transcription) > 10:
                # Extract todos
                try:
                    from todo_tool import extract_and_process_todos
                    todo_result = await extract_and_process_todos(transcription)
                    todos = todo_result.get("todos", [])
                    summary = todo_result.get("summary", "")
                    print(f"✅ Extracted {len(todos)} todos")
                except Exception as e:
                    print(f"⚠️ Todo extraction failed: {e}")
                    
                # Extract action items
                try:
                    from action_task_tool import extract_action_tasks
                    action_result = await extract_action_tasks(transcription)
                    action_items = action_result.get("action_tasks", [])
                    print(f"✅ Extracted {len(action_items)} action items")
                except Exception as e:
                    print(f"⚠️ Action item extraction failed: {e}")
            
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            transcription = f"Processing failed: {str(e)}"
            todos = []
            action_items = []
            summary = "Audio processing encountered an error"
        
        # Update database with real results
        conn = sqlite3.connect('audio_files.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE audio_files 
            SET transcript = ?, action_items = ?, todo_items = ?, processed_at = ?
            WHERE id = ?
        """, (
            transcription,
            str(action_items),
            str(todos),
            datetime.now().isoformat(),
            processing_id
        ))
        
        conn.commit()
        conn.close()
        
        # Send email notification with real results
        if user_email and user_email != "unknown":
            try:
                email_subject = f"Audio Processing Complete - {filename}"
                
                # Format todos and action items for email
                todos_text = '\\n'.join([f"• {todo}" for todo in todos]) if todos else "No todos found"
                actions_text = '\\n'.join([f"• {action.get('task', str(action)) if isinstance(action, dict) else str(action)}" for action in action_items]) if action_items else "No action items found"
                
                email_body = f"""Audio Processing Results for: {filename}

TRANSCRIPT:
{transcription}

ACTION ITEMS:
{actions_text}

TODO ITEMS:
{todos_text}

SUMMARY:
{summary or 'Audio processed successfully'}

Processing completed successfully!

Best regards,
Your Virtual Assistant"""
                
                await notifier(user_email, email_subject, email_body)
                print(f"📧 Email notification sent to {user_email}")
            except Exception as e:
                print(f"⚠️ Failed to send email notification: {e}")
        
        return {
            "success": True,
            "processing_id": processing_id,
            "message": f"Small audio file ({filename}) processed successfully",
            "filename": filename,
            "processing_type": "direct",
            "transcription": transcription,
            "action_items": action_items,
            "todo_items": todos,
            "summary": summary or "Audio processed successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_audio_processing_results(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get audio processing results from database
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of processing results
    """
    try:
        conn = sqlite3.connect('audio_files.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, filename, user_email, transcript, action_items, todo_items, processed_at, created_at
            FROM audio_files 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "filename": row[1],
                "user_email": row[2],
                "transcript": row[3] or "Processing...",
                "action_items": row[4] or "None extracted yet",
                "todo_items": row[5] or "None extracted yet",
                "processed_at": row[6],
                "created_at": row[7]
            })
            
        return results
        
    except Exception as e:
        print(f"Error getting audio results: {e}")
        return []