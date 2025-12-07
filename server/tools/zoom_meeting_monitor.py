"""
Automatic Zoom Meeting Monitor
Detects when user joins Zoom meetings and starts background recording
"""

import psutil
import time
import asyncio
import threading
from typing import Optional, List
import subprocess
import re
from datetime import datetime

class ZoomMeetingMonitor:
    def __init__(self):
        """Initialize the Zoom meeting monitor"""
        self.is_monitoring = False
        self.current_meeting_id = None
        self.recording_active = False
        self.monitor_thread = None
        
        print("🎯 Zoom Meeting Monitor initialized")
        
    def is_zoom_running(self) -> bool:
        """Check if Zoom application is running - NOT AVAILABLE ON AZURE SERVER"""
        # This method cannot detect local Zoom processes from Azure server
        # Always return False to prevent false detection
        return False
    
    def get_zoom_meeting_info(self) -> Optional[dict]:
        """Try to extract meeting information - NOT AVAILABLE ON AZURE SERVER"""
        # Process detection won't work on Azure server for local Zoom processes
        # This needs to be triggered by URL detection or webhooks instead
        return None
    
    def check_zoom_window_title(self) -> Optional[str]:
        """Check Zoom window title - NOT AVAILABLE ON AZURE SERVER"""
        # Window title detection won't work on Azure server for local windows
        # This needs to be triggered by URL detection or webhooks instead
        return None

    async def trigger_meeting_recording(self, meeting_id: str, meeting_info: dict = None):
        """Manually trigger meeting recording (called when URL is detected)"""
        try:
            if not self.recording_active:
                print(f"🎯 Auto-triggered recording for meeting: {meeting_id}")
                
                # Handle None meeting_info safely
                if meeting_info is None:
                    meeting_info = {}
                
                topic = meeting_info.get('topic', f'Meeting {meeting_id}') if meeting_info else f'Auto-detected Meeting {meeting_id}'
                await self.start_background_recording(meeting_id, topic)
                return {"success": True, "message": f"Recording started for meeting {meeting_id}", "meeting_id": meeting_id}
            else:
                return {"success": False, "message": "Recording already active", "current_meeting": self.current_meeting_id}
        except Exception as e:
            print(f"❌ Error triggering meeting recording: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "message": str(e)}
    
    async def detect_meeting_from_url(self, url: str):
        """Extract meeting ID from Zoom URL and start recording"""
        try:
            # Extract meeting ID from various Zoom URL formats
            import re
            
            # Common Zoom URL patterns
            patterns = [
                r'zoom\.us/j/(\d{9,11})',  # https://zoom.us/j/123456789
                r'zoom\.us/meeting/(\d{9,11})',  # https://zoom.us/meeting/123456789
                r'confno=(\d{9,11})',  # confno=123456789
                r'meetingId=(\d{9,11})',  # meetingId=123456789
                r'/j/(\d{9,11})',  # Short format
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url, re.IGNORECASE)
                if match:
                    meeting_id = match.group(1)
                    print(f"🔗 Detected meeting ID from URL: {meeting_id}")
                    result = await self.trigger_meeting_recording(meeting_id, {'url': url})
                    return result
            
            return {"success": False, "message": "No meeting ID found in URL"}
            
        except Exception as e:
            print(f"❌ Error detecting meeting from URL: {e}")
            return {"success": False, "message": str(e)}
    
    async def start_background_recording(self, meeting_id: str, topic: str = None):
        """Start background recording for detected meeting"""
        try:
            print(f"🎙️ Starting background recording for meeting {meeting_id}")
            
            # Import recording tools
            from .zoom_webhook_tool import start_meeting_recording
            
            # Ensure topic is never None
            if topic is None:
                topic = f'Auto-detected Meeting {meeting_id}'
            
            # Create meeting session record
            result = await start_meeting_recording({
                'meeting_id': meeting_id,
                'topic': topic,
                'start_time': datetime.now().isoformat(),
                'detected_via': 'url_detection'
            })
            
            if result.get('success'):
                self.recording_active = True
                self.current_meeting_id = meeting_id
                print(f"✅ Background recording started for meeting {meeting_id}")
            else:
                print(f"❌ Failed to start recording: {result.get('error')}")
            
        except Exception as e:
            print(f"❌ Error starting background recording: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
    
    async def stop_background_recording(self):
        """Stop background recording"""
        try:
            if self.recording_active and self.current_meeting_id:
                print(f"🛑 Stopping background recording for meeting {self.current_meeting_id}")
                
                # Import recording tools
                from .zoom_webhook_tool import stop_meeting_recording
                
                # End meeting session
                result = await stop_meeting_recording({
                    'meeting_id': self.current_meeting_id,
                    'end_time': datetime.now().isoformat()
                })
                
                if result.get('success'):
                    self.recording_active = False
                    self.current_meeting_id = None
                    print("✅ Background recording stopped")
                else:
                    print(f"❌ Failed to stop recording: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Error stopping background recording: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
    
    async def monitor_loop(self):
        """Main monitoring loop - simplified for Azure server"""
        print("🔄 Starting Zoom meeting monitor loop (Azure mode)")
        print("📋 Note: Automatic meeting detection requires URL input or webhooks on Azure")
        
        while self.is_monitoring:
            try:
                # On Azure server, we can't detect local processes
                # Just keep the monitoring active for manual triggers
                # Check every 30 seconds (reduced frequency)
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"❌ Monitor loop error: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def start_monitoring(self):
        """Start the monitoring process"""
        if not self.is_monitoring:
            self.is_monitoring = True
            
            # Start monitoring in background thread
            self.monitor_thread = threading.Thread(target=self._run_monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            print("✅ Zoom meeting monitoring started")
        else:
            print("⚠️ Monitoring already active")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        if self.is_monitoring:
            self.is_monitoring = False
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            
            print("🛑 Zoom meeting monitoring stopped")
    
    def _run_monitor_loop(self):
        """Run the async monitor loop in a thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.monitor_loop())
        except Exception as e:
            print(f"❌ Monitor thread error: {e}")

# Global monitor instance
zoom_monitor = ZoomMeetingMonitor()

def start_meeting_monitor():
    """Start the global meeting monitor"""
    zoom_monitor.start_monitoring()

def stop_meeting_monitor():
    """Stop the global meeting monitor"""
    zoom_monitor.stop_monitoring()

def get_monitor_status():
    """Get current monitoring status"""
    return {
        'is_monitoring': zoom_monitor.is_monitoring,
        'recording_active': zoom_monitor.recording_active,
        'current_meeting_id': zoom_monitor.current_meeting_id
    }