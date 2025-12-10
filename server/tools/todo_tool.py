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
        """Extract todos from text using LLM with beautiful formatting"""
        prompt = f"""
You are analyzing a business meeting transcription. Create a comprehensive, well-organized summary.

IMPORTANT CONTEXT:
- If you detect Hindi/Urdu words, treat them as Pakistani business context (not Indian)
- ALL names must be written in ENGLISH only (never use Devanagari/Arabic script)
- Provide detailed, comprehensive summary with full context
- Make output highly readable and professional

Text: {text}

Required Format:

🎯 **COMPREHENSIVE MEETING SUMMARY**
[Write a detailed 3-4 sentence summary explaining: what was the main purpose of the meeting, what key topics were discussed, what major decisions were made, and what the overall outcome/next steps are. Make this comprehensive and informative.]

👥 **RESPONSIBILITIES & COMMITMENTS**

**🔸 [English Name Only]:**
• [Detailed task/commitment with context]
• [Another specific responsibility]

**🔸 Speaker (Personal Tasks):**
• [Your specific action items]

📊 **KEY DECISIONS & OUTCOMES**
• [Important decision 1 with context]
• [Important decision 2 with reasoning]
• [Any major agreements reached]

📅 **IMPORTANT DETAILS**
• [Specific dates, deadlines, timelines]
• [Technical details, URLs, specifications]
• [Contact information, meeting schedules]

Ensure ALL content is in English and professionally formatted.
"""
        
        try:
            llm_output = await call_llm(text, prompt)
            print(f"✅ LLM Output: {llm_output}")
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return {
                "summary": f"Summary: {text[:100]}...",
                "todos": []
            }
        
        # Parse comprehensive beautiful output
        summary = ""
        todos = []
        
        # Extract comprehensive meeting summary
        if "🎯 **COMPREHENSIVE MEETING SUMMARY**" in llm_output:
            parts = llm_output.split("🎯 **COMPREHENSIVE MEETING SUMMARY**", 1)[1]
            summary_part = parts.split("👥 **", 1)[0].strip()
            summary = f"🎯 **COMPREHENSIVE MEETING SUMMARY**\n{summary_part}"
        
        # Add responsibilities section
        if "👥 **RESPONSIBILITIES & COMMITMENTS**" in llm_output:
            parts = llm_output.split("👥 **RESPONSIBILITIES & COMMITMENTS**", 1)[1]
            resp_part = parts.split("📊 **", 1)[0].strip()
            summary += f"\n\n👥 **RESPONSIBILITIES & COMMITMENTS**\n{resp_part}"
        
        # Add key decisions
        if "📊 **KEY DECISIONS & OUTCOMES**" in llm_output:
            parts = llm_output.split("📊 **KEY DECISIONS & OUTCOMES**", 1)[1]
            decisions_part = parts.split("📅 **", 1)[0].strip()
            summary += f"\n\n📊 **KEY DECISIONS & OUTCOMES**\n{decisions_part}"
        
        # Add important details
        if "📅 **IMPORTANT DETAILS**" in llm_output:
            parts = llm_output.split("📅 **IMPORTANT DETAILS**", 1)[1].strip()
            summary += f"\n\n📅 **IMPORTANT DETAILS**\n{parts}"
        
        # Extract personal todos from Speaker section
        if "**🔸 Speaker" in llm_output:
            speaker_text = llm_output.split("**🔸 Speaker", 1)[1]
            for line in speaker_text.split("\n"):
                if line.strip().startswith("•"):
                    todo = line.strip("• ").strip()
                    if todo and len(todo) > 5:
                        todos.append(todo)
                        await self.add_todo(todo)
        
        if not summary:
            summary = f"📋 Comprehensive meeting analysis: {text[:150]}..."
        
        print(f"✅ Generated comprehensive summary with {len(todos)} personal todos")
        return {
            "summary": summary,
            "todos": todos
        }
        prompt = f"""
You are an intelligent assistant that understands multiple languages including English, Hindi, Urdu, and others.
From the following transcription, identify SPECIFIC PERSONS mentioned and their RELATED TASKS, COMMITMENTS, or DISCUSSIONS.

CRITICAL INSTRUCTIONS:
1. IDENTIFY ALL PERSONS: Extract names, roles, or identifiers (e.g., "Ahmed", "Huzaifa", "CEO", "manager", "client")
2. PERSON-TASK MAPPING: For each person, list what they said, committed to, or what was discussed about them
3. SPEAKER'S PERSONAL TASKS: Extract tasks the speaker (person recording) needs to do personally
4. CONVERSATION CONTEXT: Capture who said what and when decisions were made
5. RESPONSIBILITY TRACKING: Record who is responsible for each action or decision
6. FOLLOW-UPS: Note any follow-up actions needed with specific people

ENHANCED EXAMPLES:
Input: "Ahmed said he will complete the report by Friday. I need to review it and send feedback to the client."
Output:
- Ahmed: Complete the report by Friday
- Speaker (Personal): Review Ahmed's report and send feedback to client

Input: "In today's meeting, Huzaifa discussed the backend architecture. Sarah will handle frontend design. I promised to coordinate with both teams and prepare the project timeline."
Output:
- Huzaifa: Discussed backend architecture
- Sarah: Handle frontend design  
- Speaker (Personal): Coordinate with both teams, Prepare project timeline

Input: "CEO ne kaha budget finalize karna hai. Main financial projections prepare karunga."
Output:
- CEO: Said to finalize budget
- Speaker (Personal): Prepare financial projections

Text to analyze: {text}

IMPORTANT: Focus on PERSON-SPECIFIC information and WHO is responsible for WHAT. Not generic summaries.

Output format:
Summary: [Detailed summary including WHO discussed WHAT, key decisions made by specific people, and important context with names/roles]

Key Topics: [Main subjects discussed, mentioning which person was involved in each topic]

Person-Task Mapping:
- [Person Name/Role]: [What they said, committed to, or discussed]
- [Another Person]: [Their specific tasks or commitments]
- Speaker (Personal): [Personal tasks for the speaker/recorder]

Important Details: [Specific dates, times, locations, numbers, deadlines mentioned by specific people]

Todos (Speaker's Personal Tasks):
- [personal task 1 in English]
- [personal task 2 in English]

If no personal tasks identified, say:
Todos (Speaker's Personal Tasks):
No personal tasks identified
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
        
        # Parse the output with enhanced person-task structure
        summary = ""
        key_topics = ""
        person_task_mapping = ""
        important_details = ""
        todos = []
        
        # Extract Summary
        if "Summary:" in llm_output:
            parts = llm_output.split("Summary:", 1)
            if len(parts) > 1:
                next_section = "Key Topics:" if "Key Topics:" in parts[1] else "Person-Task" if "Person-Task" in parts[1] else "Important Details:"
                summary_part = parts[1].split(next_section, 1)[0].strip()
                summary = summary_part
        
        # Extract Key Topics
        if "Key Topics:" in llm_output:
            parts = llm_output.split("Key Topics:", 1)
            if len(parts) > 1:
                next_section = "Person-Task" if "Person-Task" in parts[1] else "Important Details:"
                topics_part = parts[1].split(next_section, 1)[0].strip()
                key_topics = topics_part
        
        # Extract Person-Task Mapping
        if "Person-Task Mapping:" in llm_output:
            parts = llm_output.split("Person-Task Mapping:", 1)
            if len(parts) > 1:
                next_section = "Important Details:" if "Important Details:" in parts[1] else "Todos"
                person_part = parts[1].split(next_section, 1)[0].strip()
                person_task_mapping = person_part
        
        # Extract Important Details
        if "Important Details:" in llm_output:
            parts = llm_output.split("Important Details:", 1)
            if len(parts) > 1:
                details_part = parts[1].split("Todos", 1)[0].strip()
                important_details = details_part
        
        # Extract Todos (Speaker's Personal Tasks)
        if "Todos" in llm_output and "Speaker's Personal Tasks" in llm_output:
            todos_part = llm_output.split("Todos", 1)[1].strip()
            
            if "No personal tasks identified" not in todos_part and "No todos found" not in todos_part:
                for line in todos_part.split("\n"):
                    if line.strip().startswith("-"):
                        todo = line.strip("- ").strip()
                        if todo and "Speaker's Personal Tasks" not in todo and "Personal Tasks" not in todo:
                            todos.append(todo)
                            await self.add_todo(todo)
        
        # Create comprehensive summary with person-task context
        if not summary:
            summary = f"Analysis of voice input: {text[:100]}..."
        
        # Build detailed summary with person-task information
        detailed_summary = summary
        
        if person_task_mapping:
            detailed_summary += f"\n\n👥 Person-Task Mapping:\n{person_task_mapping}"
        
        if key_topics:
            detailed_summary += f"\n\n📋 Key Topics: {key_topics}"
        
        if important_details:
            detailed_summary += f"\n\n📌 Important Details: {important_details}"
        
        print(f"✅ Extracted {len(todos)} personal todos with person-specific context")
        return {
            "summary": detailed_summary,
            "todos": todos,
            "key_topics": key_topics,
            "person_task_mapping": person_task_mapping,
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
