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

async def extract_action_tasks(text: str) -> dict:
    """Extract actionable tasks with assignments from text"""
    prompt = f"""
You are an intelligent assistant that understands multiple languages including English, Hindi, Urdu, and others.
From the following text, extract actionable tasks that are assigned to specific people.
Each task must clearly state: what to do, who will do it, and any mentioned due date.

Examples:
- "Ahmed, please write the report by Friday" → task: Write the report, assigned_to: Ahmed, due_date: Friday
- "Mohammad, your task is to code encryption by Nov 20th" → task: Code encryption, assigned_to: Mohammad, due_date: Nov 20th
- "Aapka task hai Python program likhna" → task: Write Python program, assigned_to: User, due_date: Not specified
- "I need to buy groceries" → Not an action task (personal todo)

Text: {text}

Use your intelligence to identify tasks in any language. Translate and extract the task information.
IMPORTANT: All output must be in English only, regardless of input language.

Output format:
Action Tasks:
- task: [task description in English], assigned_to: [person name], due_date: [date or Not specified]

If no action tasks found, just say:
Action Tasks:
No actionable tasks found
"""
    
    try:
        llm_output = await call_llm(text, prompt)
        print(f"✅ LLM Output: {llm_output}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        # Very minimal fallback - just return empty if LLM completely fails
        return {"action_tasks": []}
    
    # Parse the output
    action_tasks = []
    if "Action Tasks:" in llm_output:
        tasks_part = llm_output.split("Action Tasks:", 1)[1].strip()
        
        if "No actionable tasks found" not in tasks_part:
            for line in tasks_part.split("\n"):
                if line.strip().startswith("-"):
                    line = line.strip("- ").strip()
                    # Parse "task: ..., assigned_to: ..., due_date: ..."
                    parts = line.split(", ")
                    task = "Not specified"
                    assigned_to = "Not specified" 
                    due_date = "Not specified"
                    
                    for part in parts:
                        if part.startswith("task:"):
                            task = part.replace("task:", "").strip()
                        elif part.startswith("assigned_to:"):
                            assigned_to = part.replace("assigned_to:", "").strip()
                        elif part.startswith("due_date:"):
                            due_date = part.replace("due_date:", "").strip()
                    
                    if task != "Not specified":
                        action_tasks.append({"task": task, "assigned_to": assigned_to, "due_date": due_date})
    
    print(f"✅ Extracted {len(action_tasks)} action tasks")
    return {"action_tasks": action_tasks}