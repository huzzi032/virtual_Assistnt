# server/tools/todo_tool.py
import sqlite3
import os
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def call_llm(text: str, prompt: str, max_retries: int = 3) -> str:
    """Utility to call OpenAI GPT-4.1-mini with rate limiting and retry logic"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL environment variables are required")
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    base_url,
                    headers={"api-key": api_key, "Content-Type": "application/json"},
                    json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000}
                ) as resp:
                    result = await resp.json()
                    
                    # Check for rate limit errors
                    if "error" in result and "rate limit" in result["error"]["message"].lower():
                        wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                        print(f"⚠️ Rate limit hit, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Check for other API errors
                    if "error" in result:
                        raise ValueError(f"OpenAI API Error: {result['error']['message']}")
                    
                    # Check if choices exist
                    if "choices" not in result or not result["choices"]:
                        raise ValueError(f"Unexpected API response: {result}")
                    
                    output = result["choices"][0]["message"]["content"]
                    return output
                    
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            print(f"⚠️ LLM request failed on attempt {attempt + 1}: {e}")
            await asyncio.sleep(1)  # Brief wait before retry
    
    raise Exception("Max retries exceeded for LLM call")

class TodoTool:
    def __init__(self, db_path="database.db"):
        self.db = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the todos table"""
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS todos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )""")
        conn.commit()
        conn.close()

    async def add_todo(self, item: str) -> str:
        """Add a single todo item to database"""
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("INSERT INTO todos (item) VALUES (?)", (item,))
        conn.commit()
        conn.close()
        return "added"

    async def extract_todos_from_text(self, text: str) -> dict:
        """Extract todos from text using LLM"""
        prompt = f"""
You are an intelligent assistant that understands multiple languages including English, Hindi, Urdu, and others.
From the following text, extract personal todos and reminders, and provide a comprehensive summary.

IMPORTANT GUIDELINES:
1. Extract PERSONAL tasks that the speaker themselves needs to do
2. Look for phrases like "I need to", "remind me to", "write a note", "take notes", "do this", etc.
3. If someone says "write a note on X" - this is a personal task for the speaker
4. Convert tasks assigned to others into personal reminders (e.g., "Ahmed write report" → "Follow up with Ahmed on report")
5. Look for implicit personal tasks (like taking notes, remembering information)

Examples:
- "Write a note on Qaida Azam" → Write a note on Qaida Azam
- "I need to buy groceries and call my mother" → Buy groceries, Call my mother
- "Remind me to pay the electricity bill" → Pay the electricity bill  
- "Mujhe shopping karna hai" → Go shopping
- "Ahmed, write the report" → Follow up with Ahmed on the report
- "Take notes about the meeting" → Take notes about the meeting

Text to analyze: {text}

For the input "Write a note on Qaida Azam" or similar note-taking requests, this should ALWAYS be treated as a personal todo for the speaker.

Use your intelligence to identify personal todos in any language. Translate them to English.
IMPORTANT: All output must be in English only, regardless of input language.

Output format:
Summary: [Provide a detailed summary of what was discussed, including key topics, important points, decisions made, and any context. Make it comprehensive and informative - at least 2-3 sentences covering the main content and any important details mentioned.]

Key Topics: [List the main topics or subjects discussed]

Important Details: [Any specific details, dates, names, locations, or other important information mentioned]

Todos:
- [personal todo 1 in English]
- [personal todo 2 in English]

If no personal todos found, still provide a detailed summary and say:
Todos:
No todos found
"""
        
        try:
            llm_output = await call_llm(text, prompt)
            print(f"✅ LLM Output: {llm_output}")
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            # Very minimal fallback - just return empty if LLM completely fails
            return {
                "summary": f"Summary: {text[:100]}...",
                "todos": []
            }
        
        # Parse the output
        summary = ""
        key_topics = ""
        important_details = ""
        todos = []
        
        # Enhanced parsing for detailed summary
        if "Summary:" in llm_output:
            parts = llm_output.split("Summary:", 1)
            if len(parts) > 1:
                summary_part = parts[1].split("Key Topics:", 1)[0].strip()
                summary = summary_part
        
        if "Key Topics:" in llm_output:
            parts = llm_output.split("Key Topics:", 1)
            if len(parts) > 1:
                topics_part = parts[1].split("Important Details:", 1)[0].strip()
                key_topics = topics_part
        
        if "Important Details:" in llm_output:
            parts = llm_output.split("Important Details:", 1)
            if len(parts) > 1:
                details_part = parts[1].split("Todos:", 1)[0].strip()
                important_details = details_part
        
        if "Todos:" in llm_output:
            todos_part = llm_output.split("Todos:", 1)[1].strip()
            
            if "No todos found" not in todos_part:
                for line in todos_part.split("\n"):
                    if line.strip().startswith("-"):
                        todo = line.strip("- ").strip()
                        if todo:
                            todos.append(todo)
                            await self.add_todo(todo)
        
        # Create comprehensive summary
        if not summary:
            summary = f"Analysis of voice input: {text[:100]}..."
        
        # Combine all information for detailed summary
        detailed_summary = summary
        if key_topics:
            detailed_summary += f"\n\nKey Topics Discussed: {key_topics}"
        if important_details:
            detailed_summary += f"\n\nImportant Details: {important_details}"
        
        print(f"✅ Extracted {len(todos)} todos with detailed summary")
        return {
            "summary": detailed_summary,
            "todos": todos,
            "key_topics": key_topics,
            "important_details": important_details
        }

    async def get_all_todos(self) -> list:
        """Get all todos from database"""
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("SELECT id, item, timestamp FROM todos ORDER BY timestamp DESC")
        todos = []
        for row in cur.fetchall():
            todos.append({
                "id": row[0],
                "item": row[1],
                "timestamp": row[2]
            })
        conn.close()
        return todos

    async def delete_todo(self, todo_id: int) -> bool:
        """Delete a todo by ID"""
        try:
            conn = sqlite3.connect(self.db)
            cur = conn.cursor()
            cur.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
            conn.commit()
            affected = cur.rowcount
            conn.close()
            return affected > 0
        except Exception as e:
            print(f"❌ Error deleting todo: {e}")
            return False

# Standalone functions for backward compatibility
async def extract_and_process_todos(text: str) -> dict:
    """Extract todos from text and add to database"""
    tool = TodoTool()
    return await tool.extract_todos_from_text(text)

async def todo_manager(item: str) -> str:
    """Add a todo item"""
    tool = TodoTool()
    return await tool.add_todo(item)

async def get_todos() -> list:
    """Get all todos from database"""
    tool = TodoTool()
    return await tool.get_all_todos()
