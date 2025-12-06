# server/tools/zoom_meeting_tool.py
"""
Zoom Meeting Management Tool
Handles meeting detection, URL parsing, and automatic meeting linking
"""

import re
import aiohttp
import asyncio
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv
from .zoom_oauth_tool import get_zoom_access_token

load_dotenv()

class ZoomMeetingManager:
    def __init__(self):
        """Initialize Zoom meeting manager"""
        
        # Zoom API base URL
        self.api_base_url = "https://api.zoom.us/v2"
        
        # Database
        self.db_path = "database.db"
        
        # Meeting URL patterns for detection
        self.zoom_url_patterns = [
            r'https?://[\w-]+\.zoom\.us/j/\d+',
            r'https?://us\d+web\.zoom\.us/j/\d+',
            r'https?://zoom\.us/j/\d+',
            r'https?://[\w-]+\.zoom\.us/s/\d+',
        ]
        
        print(f"🔗 Zoom Meeting Manager initialized")
        print(f"   API Base: {self.api_base_url}")
        print(f"   URL Patterns: {len(self.zoom_url_patterns)} patterns configured")

    def detect_zoom_urls(self, text: str) -> list:
        """Detect Zoom meeting URLs in text"""
        
        found_urls = []
        
        for pattern in self.zoom_url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_urls.extend(matches)
        
        # Remove duplicates while preserving order
        unique_urls = []
        for url in found_urls:
            if url not in unique_urls:
                unique_urls.append(url)
        
        if unique_urls:
            print(f"🔍 Detected {len(unique_urls)} Zoom URLs in text")
            for url in unique_urls:
                print(f"   📎 {url}")
        
        return unique_urls

    def parse_zoom_url(self, zoom_url: str) -> dict:
        """Parse Zoom URL to extract meeting information"""
        
        try:
            parsed_url = urlparse(zoom_url)
            
            # Extract meeting ID from URL path
            path_parts = parsed_url.path.split('/')
            meeting_id = None
            
            if '/j/' in parsed_url.path:
                # Join URL format: /j/123456789
                j_index = path_parts.index('j')
                if j_index + 1 < len(path_parts):
                    meeting_id = path_parts[j_index + 1].split('?')[0]  # Remove query params
            
            elif '/s/' in parsed_url.path:
                # Start URL format: /s/123456789  
                s_index = path_parts.index('s')
                if s_index + 1 < len(path_parts):
                    meeting_id = path_parts[s_index + 1].split('?')[0]
            
            # Extract query parameters
            query_params = parse_qs(parsed_url.query)
            
            # Common Zoom URL parameters
            passcode = query_params.get('pwd', [None])[0]
            waiting_room = 'uname' in query_params
            
            meeting_info = {
                "original_url": zoom_url,
                "meeting_id": meeting_id,
                "passcode": passcode,
                "domain": parsed_url.netloc,
                "has_waiting_room": waiting_room,
                "query_params": query_params,
                "is_valid": meeting_id is not None
            }
            
            print(f"🔍 Parsed Zoom URL:")
            print(f"   Meeting ID: {meeting_id}")
            print(f"   Domain: {parsed_url.netloc}")
            print(f"   Has Passcode: {'✅' if passcode else '❌'}")
            
            return meeting_info
            
        except Exception as e:
            print(f"❌ Error parsing Zoom URL {zoom_url}: {e}")
            return {
                "original_url": zoom_url,
                "meeting_id": None,
                "is_valid": False,
                "error": str(e)
            }

    async def get_meeting_info(self, meeting_id: str) -> dict:
        """Get meeting information from Zoom API"""
        
        try:
            # Get access token
            access_token = await get_zoom_access_token()
            
            # API endpoint
            url = f"{self.api_base_url}/meetings/{meeting_id}"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    
                    if response.status == 200:
                        meeting_data = await response.json()
                        
                        # Extract key information
                        meeting_info = {
                            "success": True,
                            "meeting_id": meeting_data.get('id'),
                            "uuid": meeting_data.get('uuid'),
                            "topic": meeting_data.get('topic'),
                            "type": meeting_data.get('type'),
                            "status": meeting_data.get('status'),
                            "start_time": meeting_data.get('start_time'),
                            "duration": meeting_data.get('duration'),
                            "timezone": meeting_data.get('timezone'),
                            "join_url": meeting_data.get('join_url'),
                            "start_url": meeting_data.get('start_url'),
                            "host_id": meeting_data.get('host_id'),
                            "created_at": meeting_data.get('created_at')
                        }
                        
                        # Store in database
                        await self._store_meeting_info(meeting_info)
                        
                        print(f"✅ Meeting info retrieved: {meeting_info.get('topic')}")
                        return meeting_info
                    
                    elif response.status == 404:
                        return {
                            "success": False,
                            "error": "Meeting not found",
                            "meeting_id": meeting_id
                        }
                    
                    elif response.status == 401:
                        return {
                            "success": False,
                            "error": "Authentication required - please reconnect Zoom",
                            "meeting_id": meeting_id
                        }
                    
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"API error {response.status}: {error_text}",
                            "meeting_id": meeting_id
                        }
                        
        except Exception as e:
            print(f"❌ Error fetching meeting info for {meeting_id}: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "meeting_id": meeting_id
            }

    async def _store_meeting_info(self, meeting_info: dict):
        """Store meeting information in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Insert or update meeting info
            cur.execute("""
                INSERT OR REPLACE INTO zoom_meetings
                (meeting_id, meeting_uuid, topic, join_url, start_url, start_time, duration, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                meeting_info.get('meeting_id'),
                meeting_info.get('uuid'),
                meeting_info.get('topic'),
                meeting_info.get('join_url'),
                meeting_info.get('start_url'),
                meeting_info.get('start_time'),
                meeting_info.get('duration'),
                meeting_info.get('status')
            ))
            
            conn.commit()
            conn.close()
            
            print(f"💾 Meeting info stored: {meeting_info.get('meeting_id')}")
            
        except Exception as e:
            print(f"❌ Error storing meeting info: {e}")

    async def process_text_for_meetings(self, text: str) -> dict:
        """Process text to detect and link Zoom meetings"""
        
        # Detect Zoom URLs
        zoom_urls = self.detect_zoom_urls(text)
        
        if not zoom_urls:
            return {
                "meetings_detected": False,
                "meetings": [],
                "message": "No Zoom meeting URLs detected in text"
            }
        
        detected_meetings = []
        oauth_required = False
        
        for url in zoom_urls:
            # Parse URL
            url_info = self.parse_zoom_url(url)
            
            if url_info.get('is_valid') and url_info.get('meeting_id'):
                meeting_id = url_info['meeting_id']
                
                # Try to get meeting details from API (requires OAuth)
                meeting_details = await self.get_meeting_info(meeting_id)
                
                # Always include basic URL info
                combined_info = {
                    **url_info,
                    "detected_from_url": url,
                    "recording_ready": False
                }
                
                if meeting_details.get('success'):
                    # OAuth is working, add full meeting details
                    combined_info.update(meeting_details)
                    combined_info["recording_ready"] = True
                    combined_info["message"] = "Meeting detected and ready for recording"
                else:
                    # OAuth not complete, but still provide basic info
                    oauth_required = True
                    combined_info.update({
                        "success": True,  # URL parsing was successful
                        "topic": f"Meeting {meeting_id}",
                        "message": "Meeting detected but OAuth authorization required for recording",
                        "oauth_required": True,
                        "error_details": meeting_details.get('error', 'OAuth required')
                    })
                
                detected_meetings.append(combined_info)
            else:
                # Store invalid URLs for debugging
                url_info["detected_from_url"] = url
                detected_meetings.append(url_info)
        
        result = {
            "meetings_detected": len(detected_meetings) > 0,
            "meetings_count": len(detected_meetings),
            "meetings": detected_meetings,
            "oauth_required": oauth_required,
            "message": f"Detected {len(detected_meetings)} Zoom meeting(s). {'OAuth authorization required for recording.' if oauth_required else 'Ready for recording.'}"
        }
        
        if detected_meetings:
            print(f"🎯 Processed {len(detected_meetings)} Zoom meetings from text")
            if oauth_required:
                print("⚠️ OAuth authorization required for full meeting access")
        
        return result

    def get_stored_meetings(self, limit: int = 10) -> list:
        """Get recently stored meetings"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT meeting_id, topic, join_url, start_time, status, created_at
                FROM zoom_meetings
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            results = cur.fetchall()
            conn.close()
            
            meetings = []
            for row in results:
                meetings.append({
                    "meeting_id": row[0],
                    "topic": row[1],
                    "join_url": row[2],
                    "start_time": row[3],
                    "status": row[4],
                    "created_at": row[5]
                })
            
            return meetings
            
        except Exception as e:
            print(f"❌ Error fetching stored meetings: {e}")
            return []

    async def enable_real_time_processing(self, meeting_id: str) -> dict:
        """Enable real-time processing for a meeting"""
        
        try:
            # Update database to enable real-time processing
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE zoom_meetings 
                SET real_time_processing = 1
                WHERE meeting_id = ?
            """, (meeting_id,))
            
            # Also update session if exists
            cur.execute("""
                INSERT OR REPLACE INTO zoom_meeting_sessions
                (meeting_id, real_time_processing, auto_summary)
                VALUES (?, 1, 1)
            """, (meeting_id,))
            
            conn.commit()
            conn.close()
            
            print(f"⚡ Real-time processing enabled for meeting: {meeting_id}")
            
            return {
                "success": True,
                "meeting_id": meeting_id,
                "real_time_enabled": True,
                "message": "Real-time processing enabled for this meeting"
            }
            
        except Exception as e:
            print(f"❌ Error enabling real-time processing: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global instance
zoom_meeting_manager = ZoomMeetingManager()

# Convenience functions
def detect_zoom_meetings(text: str) -> list:
    """Detect Zoom meeting URLs in text"""
    return zoom_meeting_manager.detect_zoom_urls(text)

def parse_zoom_meeting_url(url: str) -> dict:
    """Parse Zoom URL"""
    return zoom_meeting_manager.parse_zoom_url(url)

async def get_zoom_meeting_details(meeting_id: str) -> dict:
    """Get meeting details from API"""
    return await zoom_meeting_manager.get_meeting_info(meeting_id)

async def process_text_for_zoom_meetings(text: str) -> dict:
    """Process text for Zoom meeting detection"""
    return await zoom_meeting_manager.process_text_for_meetings(text)

def get_recent_zoom_meetings(limit: int = 10) -> list:
    """Get recent Zoom meetings"""
    return zoom_meeting_manager.get_stored_meetings(limit)

async def enable_meeting_real_time_processing(meeting_id: str) -> dict:
    """Enable real-time processing for meeting"""
    return await zoom_meeting_manager.enable_real_time_processing(meeting_id)

if __name__ == "__main__":
    # Test meeting detection
    manager = ZoomMeetingManager()
    
    test_text = """
    Join our meeting at https://zoom.us/j/123456789?pwd=abc123
    Also check https://us02web.zoom.us/j/987654321
    """
    
    urls = manager.detect_zoom_urls(test_text)
    print(f"Detected URLs: {urls}")
    
    for url in urls:
        info = manager.parse_zoom_url(url)
        print(f"Parsed: {info}")