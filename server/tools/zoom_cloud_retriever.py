"""
Zoom Cloud Meeting Retrieval Tool
Retrieves completed meetings from Zoom cloud and processes recordings
"""

import aiohttp
import asyncio
import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class ZoomCloudRetriever:
    def __init__(self):
        self.base_url = "https://api.zoom.us/v2"
        self.access_token = None
        
    def init_database(self):
        """Initialize zoom meetings database"""
        conn = sqlite3.connect('zoom_meetings.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zoom_meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT UNIQUE,
                uuid TEXT,
                topic TEXT,
                start_time TEXT,
                end_time TEXT,
                duration INTEGER,
                recording_url TEXT,
                transcript TEXT,
                summary TEXT,
                action_items TEXT,
                todo_items TEXT,
                processed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
    async def get_access_token(self) -> Optional[str]:
        """Get valid access token from zoom oauth"""
        try:
            from zoom_oauth_tool import zoom_oauth_manager
            token_info = await zoom_oauth_manager.get_valid_access_token()
            if isinstance(token_info, dict):
                access_token = token_info.get('access_token')
                if access_token:
                    self.access_token = access_token
                    return self.access_token
            return None
        except Exception as e:
            print(f"❌ Error getting access token: {e}")
            return None
    
    async def get_completed_meetings(self, days_back: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve completed meetings from Zoom
        
        Args:
            days_back: Number of days to look back for meetings
            
        Returns:
            List of completed meetings
        """
        try:
            if not self.access_token:
                await self.get_access_token()
                
            if not self.access_token:
                return []
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'type': 'past',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'page_size': 30
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/users/me/meetings"
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        meetings = data.get('meetings', [])
                        
                        print(f"📋 Found {len(meetings)} completed meetings")
                        return meetings
                    else:
                        print(f"❌ Error fetching meetings: {response.status}")
                        return []
                        
        except Exception as e:
            print(f"❌ Error retrieving meetings: {e}")
            return []
    
    async def get_meeting_recordings(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """
        Get recordings for a specific meeting
        
        Args:
            meeting_id: Zoom meeting ID
            
        Returns:
            Recording information or None
        """
        try:
            if not self.access_token:
                await self.get_access_token()
                
            if not self.access_token:
                return None
                
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/meetings/{meeting_id}/recordings"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 404:
                        print(f"ℹ️ No recordings found for meeting {meeting_id}")
                        return None
                    else:
                        print(f"❌ Error fetching recordings for {meeting_id}: {response.status}")
                        return None
                        
        except Exception as e:
            print(f"❌ Error getting recordings: {e}")
            return None
    
    async def download_recording(self, download_url: str, access_token: str, meeting_id: str) -> Optional[str]:
        """
        Download recording file
        
        Args:
            download_url: URL to download the recording
            access_token: Access token for download
            meeting_id: Meeting ID for filename
            
        Returns:
            Path to downloaded file or None
        """
        try:
            # Create recordings directory
            recordings_dir = "recordings"
            os.makedirs(recordings_dir, exist_ok=True)
            
            headers = {
                'Authorization': f'Bearer {access_token}'
            }
            
            filename = f"meeting_{meeting_id}_{int(datetime.now().timestamp())}.mp4"
            file_path = os.path.join(recordings_dir, filename)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url, headers=headers) as response:
                    if response.status == 200:
                        with open(file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        print(f"📁 Downloaded recording: {file_path}")
                        return file_path
                    else:
                        print(f"❌ Download failed: {response.status}")
                        return None
                        
        except Exception as e:
            print(f"❌ Error downloading recording: {e}")
            return None
    
    async def process_meeting_for_cloud_recording(self, meeting_id: str) -> Dict[str, Any]:
        """
        Process a specific meeting to get and process its cloud recording
        
        Args:
            meeting_id: Zoom meeting ID
            
        Returns:
            Processing result
        """
        try:
            print(f"🔄 Processing meeting {meeting_id} for cloud recording...")
            
            self.init_database()
            
            # Check if already processed
            conn = sqlite3.connect('zoom_meetings.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM zoom_meetings WHERE meeting_id = ?", (meeting_id,))
            existing = cursor.fetchone()
            
            if existing:
                conn.close()
                return {
                    "success": True,
                    "message": f"Meeting {meeting_id} already processed",
                    "meeting_id": meeting_id
                }
            
            # Get recordings
            recording_data = await self.get_meeting_recordings(meeting_id)
            
            if not recording_data:
                conn.close()
                return {
                    "success": False,
                    "error": f"No recordings found for meeting {meeting_id}"
                }
            
            # Extract recording info
            recording_files = recording_data.get('recording_files', [])
            audio_file = None
            
            for file in recording_files:
                if file.get('file_type') in ['MP4', 'M4A', 'MP3']:
                    audio_file = file
                    break
            
            if not audio_file:
                conn.close()
                return {
                    "success": False,
                    "error": f"No audio recording found for meeting {meeting_id}"
                }
            
            # Download the recording
            download_url = audio_file.get('download_url')
            if download_url and self.access_token:
                file_path = await self.download_recording(download_url, self.access_token, meeting_id)
                
                if file_path:
                    # Store meeting info
                    cursor.execute("""
                        INSERT INTO zoom_meetings 
                        (meeting_id, uuid, topic, start_time, recording_url, processed_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        meeting_id,
                        recording_data.get('uuid', ''),
                        recording_data.get('topic', 'Zoom Meeting'),
                        recording_data.get('start_time', ''),
                        file_path,
                        datetime.now().isoformat()
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                    # Process the downloaded audio file
                    from audio_processor import process_uploaded_audio
                    audio_result = await process_uploaded_audio(
                        file_path, 
                        f"zoom_meeting_{meeting_id}.mp4",
                        os.path.getsize(file_path),
                        "zoom_cloud_processing@system.local"
                    )
                    
                    return {
                        "success": True,
                        "message": f"Meeting {meeting_id} downloaded and queued for processing",
                        "meeting_id": meeting_id,
                        "file_path": file_path,
                        "audio_processing": audio_result
                    }
            elif not self.access_token:
                conn.close()
                return {
                    "success": False,
                    "error": "No valid access token available for download"
                }
            
            conn.close()
            return {
                "success": False,
                "error": f"Failed to download recording for meeting {meeting_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def check_for_new_completed_meetings(self) -> Dict[str, Any]:
        """
        Check for new completed meetings and process them
        
        Returns:
            Summary of processing results
        """
        try:
            print("🔍 Checking for new completed meetings...")
            
            meetings = await self.get_completed_meetings()
            processed_count = 0
            results = []
            
            for meeting in meetings:
                meeting_id = str(meeting.get('id', ''))
                if meeting_id:
                    result = await self.process_meeting_for_cloud_recording(meeting_id)
                    results.append(result)
                    if result.get('success'):
                        processed_count += 1
            
            return {
                "success": True,
                "total_meetings": len(meetings),
                "processed_count": processed_count,
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Global instance
zoom_cloud_retriever = ZoomCloudRetriever()

# Convenience functions for API endpoints
async def get_completed_zoom_meetings(days_back: int = 1) -> List[Dict[str, Any]]:
    """Get completed zoom meetings"""
    return await zoom_cloud_retriever.get_completed_meetings(days_back)

async def process_zoom_meeting_recording(meeting_id: str) -> Dict[str, Any]:
    """Process a specific zoom meeting recording"""
    return await zoom_cloud_retriever.process_meeting_for_cloud_recording(meeting_id)

async def check_new_zoom_meetings() -> Dict[str, Any]:
    """Check for new completed meetings and process them"""
    return await zoom_cloud_retriever.check_for_new_completed_meetings()