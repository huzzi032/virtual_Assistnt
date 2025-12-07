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
        """Check if Zoom application is running"""
        for process in psutil.process_iter(['name']):
            try:
                if 'zoom' in process.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    
    def get_zoom_meeting_info(self) -> Optional[dict]:
        """Try to extract meeting information from running Zoom process"""
        try:
            # Look for Zoom processes with command line arguments
            for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'zoom' in process.info['name'].lower():
                        cmdline = ' '.join(process.info['cmdline']) if process.info['cmdline'] else ''
                        
                        # Look for meeting ID patterns in command line
                        meeting_id_match = re.search(r'(?:confno=|meeting.*?id[=:]?)(\d{9,11})', cmdline, re.IGNORECASE)
                        if meeting_id_match:
                            return {
                                'meeting_id': meeting_id_match.group(1),
                                'detected_at': datetime.now(),
                                'process_name': process.info['name']
                            }
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            print(f"❌ Error getting meeting info: {e}")
            
        return None
    
    def check_zoom_window_title(self) -> Optional[str]:
        """Check Zoom window title for meeting information (Windows specific)"""
        try:
            import win32gui
            
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if 'zoom' in title.lower() and 'meeting' in title.lower():
                        windows.append(title)
                return True
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            for window_title in windows:
                # Look for meeting ID in window title
                meeting_id_match = re.search(r'(\d{9,11})', window_title)
                if meeting_id_match:
                    return meeting_id_match.group(1)
                    
        except ImportError:
            # win32gui not available, skip window title checking
            pass
        except Exception as e:
            print(f"❌ Error checking window titles: {e}")
            
        return None
    
    async def start_background_recording(self, meeting_id: str):
        """Start background recording for detected meeting"""
        try:
            print(f"🎙️ Starting background recording for meeting {meeting_id}")
            
            # Import recording tools
            from .zoom_webhook_tool import zoom_webhook_handler
            
            # Create meeting session record
            await zoom_webhook_handler._handle_meeting_started({
                'meeting_id': meeting_id,
                'topic': f'Auto-detected Meeting {meeting_id}',
                'start_time': datetime.now().isoformat(),
                'detected_via': 'process_monitor'
            })
            
            self.recording_active = True
            self.current_meeting_id = meeting_id
            
            print(f"✅ Background recording started for meeting {meeting_id}")
            
        except Exception as e:
            print(f"❌ Error starting background recording: {e}")
    
    async def stop_background_recording(self):
        """Stop background recording"""
        try:
            if self.recording_active and self.current_meeting_id:
                print(f"🛑 Stopping background recording for meeting {self.current_meeting_id}")
                
                # Import recording tools
                from .zoom_webhook_tool import zoom_webhook_handler
                
                # End meeting session
                await zoom_webhook_handler._handle_meeting_ended({
                    'meeting_id': self.current_meeting_id,
                    'end_time': datetime.now().isoformat()
                })
                
                self.recording_active = False
                self.current_meeting_id = None
                
                print("✅ Background recording stopped")
                
        except Exception as e:
            print(f"❌ Error stopping background recording: {e}")
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        print("🔄 Starting Zoom meeting monitor loop")
        
        while self.is_monitoring:
            try:
                # Check if Zoom is running
                zoom_running = self.is_zoom_running()
                
                if zoom_running and not self.recording_active:
                    # Zoom is running but we're not recording - check for meeting
                    meeting_info = self.get_zoom_meeting_info()
                    meeting_id_from_title = self.check_zoom_window_title()
                    
                    detected_meeting_id = None
                    if meeting_info:
                        detected_meeting_id = meeting_info['meeting_id']
                    elif meeting_id_from_title:
                        detected_meeting_id = meeting_id_from_title
                    
                    if detected_meeting_id:
                        print(f"🎯 Detected Zoom meeting: {detected_meeting_id}")
                        await self.start_background_recording(detected_meeting_id)
                        
                elif not zoom_running and self.recording_active:
                    # Zoom stopped but we're still recording - stop recording
                    print("🔍 Zoom closed, stopping recording")
                    await self.stop_background_recording()
                
                # Check every 5 seconds
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"❌ Monitor loop error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
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