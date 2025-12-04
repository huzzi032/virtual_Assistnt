# server/tools/calendar_tool.py
import os
import re
import aiohttp
import asyncio
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Add proper path handling for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

class CalendarTool:
    def __init__(self):
        # Mobile calendar tool - no API key needed, uses OAuth2
        pass

    async def extract_date_and_task(self, text: str) -> dict:
        """Extract date and task information from user text using LLM"""
        try:
            from llm_tool import call_openai_llm
        except ImportError:
            try:
                from tools.llm_tool import call_openai_llm
            except ImportError:
                from server.tools.llm_tool import call_openai_llm
        from datetime import datetime, timedelta
        
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        prompt = f"""
        Analyze this text and extract the task/work the user wants to schedule and the date they mentioned.
        
        Text: "{text}"
        
        Today is {today}. Tomorrow is {tomorrow}.
        
        Convert relative dates to exact YYYY-MM-DD format:
        - "tomorrow" = {tomorrow}
        - "next Monday" = calculate the exact date of next Monday
        - "Friday" = calculate this Friday's date (if today is before Friday) or next Friday
        - "December 15" = 2025-12-15
        - "next week" = pick a date next week (like next Monday)
        
        Please respond in this exact format:
        Task: [what the user wants to do]
        Date: [date in YYYY-MM-DD format ONLY]
        Time: [time in HH:MM format, use 09:00 if not specified]
        
        IMPORTANT: 
        - Date must be in YYYY-MM-DD format (like 2025-11-11)
        - If no specific date is mentioned, use "Not specified" for Date
        - If no specific task is mentioned, use "Not specified" for Task
        - Convert time formats like "2 PM" to "14:00"
        """
        
        try:
            llm_output = await call_openai_llm(text, prompt)
            
            # Parse the LLM output
            task = "Work/Task"
            date = None
            time = "09:00"
            
            for line in llm_output.split('\n'):
                line = line.strip()
                if line.startswith('Task:'):
                    task_text = line.replace('Task:', '').strip()
                    if task_text != "Not specified" and task_text:
                        task = task_text
                elif line.startswith('Date:'):
                    date_str = line.replace('Date:', '').strip()
                    if date_str != "Not specified" and date_str:
                        # Validate date format
                        try:
                            datetime.strptime(date_str, '%Y-%m-%d')
                            date = date_str
                        except ValueError:
                            print(f"Invalid date format from LLM: {date_str}")
                            date = None
                elif line.startswith('Time:'):
                    time_str = line.replace('Time:', '').strip()
                    if time_str:
                        # Validate time format
                        try:
                            datetime.strptime(time_str, '%H:%M')
                            time = time_str
                        except ValueError:
                            print(f"Invalid time format from LLM: {time_str}, using default")
                            time = "09:00"
            
            return {
                "task": task,
                "date": date,
                "time": time,
                "success": date is not None and date != "Not specified"
            }
            
        except Exception as e:
            print(f"Error extracting date and task: {e}")
            return {"task": "Work/Task", "date": None, "time": "09:00", "success": False}

    async def create_simple_calendar_event(self, task: str, date: str, time: str = "09:00", duration_hours: int = 1) -> dict:
        """Create a calendar event - both locally and in Google Calendar"""
        try:
            # Parse the date and time
            event_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            end_datetime = event_datetime + timedelta(hours=duration_hours)
            
            # Store the calendar event in our database first
            import sqlite3
            conn = sqlite3.connect("database.db")
            cur = conn.cursor()
            
            # Create calendar_events table if it doesn't exist
            cur.execute("""CREATE TABLE IF NOT EXISTS calendar_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            task TEXT,
                            event_date DATE,
                            event_time TIME,
                            duration_hours INTEGER,
                            google_event_id TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )""")
            
            # Insert the event
            cur.execute("""INSERT INTO calendar_events 
                           (task, event_date, event_time, duration_hours) 
                           VALUES (?, ?, ?, ?)""", 
                        (task, date, time, duration_hours))
            
            event_id = cur.lastrowid
            conn.commit()
            
            # Try to create Google Calendar event
            google_event_id = None
            google_success = False
            google_message = ""
            
            try:
                try:
                    from mobile_calendar_auth import get_calendar_service_mobile, is_mobile_authenticated
                except ImportError:
                    try:
                        from tools.mobile_calendar_auth import get_calendar_service_mobile, is_mobile_authenticated
                    except ImportError:
                        from server.tools.mobile_calendar_auth import get_calendar_service_mobile, is_mobile_authenticated
                
                # Check if we have authentication
                if not is_mobile_authenticated():
                    print("🔐 Google Calendar not authenticated")
                    google_message = "Google Calendar authentication required. Please visit the frontend and use the 'Authenticate Google Calendar' option."
                else:
                    print("🔐 Google Calendar authenticated, creating event...")
                    service = get_calendar_service_mobile()
                    if service:
                        # Create event data for Google Calendar
                        start_time = event_datetime.isoformat()
                        end_time = end_datetime.isoformat()
                        
                        # Get user's timezone (default to UTC for now)
                        timezone = 'UTC'
                        
                        event_data = {
                            'summary': task,
                            'description': f'Created by Voice Assistant on {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                            'start': {
                                'dateTime': start_time,
                                'timeZone': timezone,
                            },
                            'end': {
                                'dateTime': end_time,
                                'timeZone': timezone,
                            },
                        }
                        
                        # Create the event in Google Calendar
                        event = service.events().insert(calendarId='primary', body=event_data).execute()
                        google_event_id = event.get('id')
                        google_success = True
                        google_message = f"Event created in Google Calendar: {event.get('htmlLink', 'No link available')}"
                        
                        # Update database with Google event ID
                        cur.execute("""UPDATE calendar_events 
                                       SET google_event_id = ? 
                                       WHERE id = ?""", 
                                    (google_event_id, event_id))
                        conn.commit()
                    else:
                        google_message = "Failed to get Google Calendar service"
                        
            except Exception as google_error:
                google_message = f"Google Calendar error: {str(google_error)}"
                print(f"Google Calendar integration error: {google_error}")
            
            conn.close()
            
            return {
                "success": True,
                "message": f"Calendar event created: '{task}' scheduled for {date} at {time}",
                "event_id": event_id,
                "task": task,
                "date": date,
                "time": time,
                "google_calendar": {
                    "success": google_success,
                    "message": google_message,
                    "event_id": google_event_id
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating calendar event: {str(e)}",
                "event_data": None
            }

    async def process_calendar_request(self, text: str) -> dict:
        """Main function to process calendar requests from voice input"""
        # Extract date and task from the text
        extraction_result = await self.extract_date_and_task(text)
        
        if not extraction_result["success"]:
            return {
                "success": False,
                "message": "Could not extract a valid date from the request",
                "extracted": extraction_result
            }
        
        # Create calendar event
        calendar_result = await self.create_simple_calendar_event(
            extraction_result["task"],
            extraction_result["date"],
            extraction_result["time"]
        )
        
        return {
            "success": calendar_result["success"],
            "message": calendar_result["message"],
            "task": extraction_result["task"],
            "scheduled_date": extraction_result["date"],
            "scheduled_time": extraction_result["time"],
            "event_id": calendar_result.get("event_id"),
            "calendar_response": calendar_result
        }

    async def get_upcoming_events(self, days: int = 7) -> dict:
        """Get upcoming calendar events"""
        try:
            import sqlite3
            conn = sqlite3.connect("database.db")
            cur = conn.cursor()
            
            # Get events for the next 'days' days
            cur.execute("""SELECT id, task, event_date, event_time, duration_hours 
                           FROM calendar_events 
                           WHERE event_date >= date('now') 
                           AND event_date <= date('now', '+{} days')
                           ORDER BY event_date, event_time""".format(days))
            
            events = []
            for row in cur.fetchall():
                events.append({
                    "id": row[0],
                    "task": row[1],
                    "date": row[2],
                    "time": row[3],
                    "duration": row[4]
                })
            
            conn.close()
            
            return {
                "success": True,
                "events": events,
                "count": len(events)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting events: {str(e)}",
                "events": []
            }

    async def run(self, args):
        """MCP tool interface"""
        text = args.get("text", "")
        return await self.process_calendar_request(text)


# Standalone function for use in other parts of the system
async def handle_calendar_request(text: str) -> dict:
    """Process calendar requests from voice input"""
    calendar_tool = CalendarTool()
    return await calendar_tool.process_calendar_request(text)


# Function to check if text contains calendar-related intent
def contains_calendar_intent(text: str) -> bool:
    """Check if text contains specific calendar/event creation intent"""
    text_lower = text.lower()
    
    # Specific date indicators
    date_patterns = [
        r'\b(tomorrow|today|yesterday)\b',
        r'\b\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(next|this)\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b\d{1,2}:\d{2}\b',  # Time patterns
        r'\b(at|on|by)\s+\d{1,2}',  # at 3, on 5th, by 10
    ]
    
    # Event-specific keywords
    event_keywords = [
        'meeting', 'appointment', 'game', 'match', 'event', 'conference',
        'deadline', 'reminder', 'schedule', 'calendar', 'book', 'reserve'
    ]
    
    # Check for date patterns
    import re
    has_date = any(re.search(pattern, text_lower) for pattern in date_patterns)
    
    # Check for event keywords
    has_event_keyword = any(keyword in text_lower for keyword in event_keywords)
    
    # Must have both date and event indicators for calendar intent
    return has_date and has_event_keyword