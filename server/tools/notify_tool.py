# server/tools/notify_tool.py
import smtplib, os, sys
from email.mime.text import MIMEText
import sqlite3
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Add proper path handling for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

try:
    from action_task_tool import extract_action_tasks
except ImportError:
    try:
        from tools.action_task_tool import extract_action_tasks
    except ImportError:
        from server.tools.action_task_tool import extract_action_tasks

async def openai_llm(text: str) -> str:
    """Summarize conversation and extract todos"""
    prompt = f"""Analyze this text and extract any tasks, todos, or actionable items mentioned. Also provide a brief summary.

Text: {text}

Please respond in this exact format:
Summary: [brief summary of the text]
Todos:
- [task 1]
- [task 2]
- [etc]

If no todos are found, just say "No todos found" after Todos:"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL environment variables are required")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            base_url,
            headers={"api-key": api_key, "Content-Type": "application/json"},
            json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 800}
        ) as resp:
            result = await resp.json()
            
            # Check for API errors
            if "error" in result:
                raise ValueError(f"OpenAI API Error: {result['error']['message']}")
            
            # Check if choices exist
            if "choices" not in result or not result["choices"]:
                raise ValueError(f"Unexpected API response: {result}")
            
            output = result["choices"][0]["message"]["content"]
    return output

async def send_periodic_email():
    receiver = os.getenv("RECEIVER_EMAIL")
    if not receiver:
        print("RECEIVER_EMAIL not set, skipping periodic emails")
        return
    while True:
        await asyncio.sleep(3 * 3600)  # 3 hours
        try:
            conn = sqlite3.connect("database.db")
            cur = conn.cursor()
            
            # Get voice inputs grouped by hourly sessions from last 3 hours
            cur.execute("""
                SELECT text, timestamp, 
                       strftime('%Y-%m-%d %H:00:00', timestamp) as session_hour
                FROM voice_inputs 
                WHERE timestamp > datetime('now', '-3 hours')
                ORDER BY timestamp
            """)
            voice_data = cur.fetchall()
            
            # Get todos from DB with timestamps
            cur.execute("""
                SELECT item, timestamp,
                       strftime('%Y-%m-%d %H:00:00', timestamp) as session_hour
                FROM todos 
                WHERE timestamp > datetime('now', '-3 hours')
                ORDER BY timestamp
            """)
            todo_data = cur.fetchall()
            conn.close()
            
            # Group data by session hours
            sessions = {}
            
            # Process voice inputs by session
            for text, timestamp, session_hour in voice_data:
                if session_hour not in sessions:
                    sessions[session_hour] = {
                        'voice_inputs': [],
                        'todos': [],
                        'summary': '',
                        'extracted_todos': [],
                        'action_tasks': []
                    }
                sessions[session_hour]['voice_inputs'].append((text, timestamp))
            
            # Process todos by session
            for todo, timestamp, session_hour in todo_data:
                if session_hour not in sessions:
                    sessions[session_hour] = {
                        'voice_inputs': [],
                        'todos': [],
                        'summary': '',
                        'extracted_todos': [],
                        'action_tasks': []
                    }
                sessions[session_hour]['todos'].append((todo, timestamp))
            
            # Process each session separately
            for session_hour in sorted(sessions.keys()):
                session = sessions[session_hour]
                
                if session['voice_inputs']:
                    # Combine all voice inputs for this session
                    session_text = " ".join([text for text, _ in session['voice_inputs']])
                    
                    # Generate summary and extract todos for this session
                    llm_output = await openai_llm(session_text)
                    if "Todos:" in llm_output:
                        summary_part, todos_part = llm_output.split("Todos:", 1)
                        session['summary'] = summary_part.replace("Summary:", "").strip()
                        session['extracted_todos'] = [
                            line.strip("- ").strip() 
                            for line in todos_part.strip().split("\n") 
                            if line.strip().startswith("-") and line.strip() != "- No todos found"
                        ]
                    else:
                        session['summary'] = llm_output.strip()
                    
                    # Extract action tasks for this session
                    action_result = await extract_action_tasks(session_text)
                    session['action_tasks'] = action_result["action_tasks"]
            
            # Generate email body with properly formatted sessions
            if sessions:
                subject = "3-Hour Voice Agent Summary - Session Report"
                body_parts = [
                    "VOICE AGENT SESSION REPORT",
                    f"Report Period: Last 3 Hours",
                    f"Total Sessions: {len(sessions)}",
                    "",
                ]
                
                session_count = 1
                for session_hour in sorted(sessions.keys()):
                    session = sessions[session_hour]
                    body_parts.append(f"SESSION {session_count} - {session_hour}")
                    body_parts.append("=" * 50)
                    
                    # Session Summary
                    if session['summary']:
                        body_parts.append("SUMMARY:")
                        body_parts.append(f"   {session['summary']}")
                        body_parts.append("")
                    
                    # Action Tasks for this session
                    if session['action_tasks']:
                        body_parts.append("ACTION TASKS:")
                        for task in session['action_tasks']:
                            body_parts.append(f"   - {task['task']}")
                            body_parts.append(f"     Assigned to: {task['assigned_to']}")
                            body_parts.append(f"     Due date: {task['due_date']}")
                        body_parts.append("")
                    
                    # All todos for this session (both extracted and manual)
                    all_session_todos = []
                    
                    # Add extracted todos
                    for todo in session['extracted_todos']:
                        all_session_todos.append(todo)
                    
                    # Add manual todos
                    for todo, timestamp in session['todos']:
                        all_session_todos.append(todo)
                    
                    if all_session_todos:
                        body_parts.append("TODO LIST:")
                        for todo in all_session_todos:
                            body_parts.append(f"   - {todo}")
                        body_parts.append("")
                    
                    body_parts.append("-" * 50)
                    body_parts.append("")
                    session_count += 1
                
                body = "\n".join(body_parts)
                await notifier(receiver, subject, body)
                print("Periodic email sent with session-based format")
            else:
                # No activity in the last 3 hours
                subject = "3-Hour Voice Agent Summary - No Activity"
                body = "No voice agent activity detected in the last 3 hours."
                await notifier(receiver, subject, body)
                print("Periodic email sent - no activity")
                
        except Exception as e:
            print(f"Error sending periodic email: {e}")

async def notifier(to: str, subject: str = "Voice Agent Notification", body: str = "No message provided") -> str:
    """Send notification"""
    gmail_user = os.getenv("GMAIL_USER")
    gmail_pass = os.getenv("GMAIL_PASS")
    if not gmail_user or not gmail_pass:
        raise ValueError("GMAIL_USER and GMAIL_PASS environment variables must be set.")
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_server:
        smtp_server.login(gmail_user, gmail_pass)
        smtp_server.sendmail(gmail_user, [to], msg.as_string())
    return "sent"

class NotificationTool:
    async def run(self, args):
        to_email = args.get("to")
        subject = args.get("subject", "Voice Agent Notification")
        body = args.get("body", "No message provided")

        gmail_user = os.getenv("GMAIL_USER")
        gmail_pass = os.getenv("GMAIL_PASS")
        if not gmail_user or not gmail_pass:
            raise ValueError("GMAIL_USER and GMAIL_PASS environment variables must be set.")
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = gmail_user
        msg["To"] = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, [to_email], msg.as_string())

        return {"status": "sent", "to": to_email}
