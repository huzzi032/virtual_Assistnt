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
    """Extract actionable tasks organized by person from transcription"""
    prompt = f"""
Extract actionable tasks from this business meeting. 

IMPORTANT:
- Write ALL names in ENGLISH only (never use Hindi/Urdu script)
- If detecting Urdu/Hindi, treat as Pakistani business context
- Focus on clear, specific action items with deadlines
- Professional, readable format

Text: {text}

Format:

📋 **ACTION ITEMS**

**🔹 [English Name]:**
• [Specific task with deadline if mentioned]
• [Another clear action item]

**🔹 Speaker:**
• [Your action items]

Only include clear, actionable tasks. If none found: "No specific action items identified."
"""
    
    try:
        llm_output = await call_llm(text, prompt)
        print(f"✅ Action Tasks Output: {llm_output}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"action_tasks": []}
    
    # Parse organized action tasks
    action_tasks = []
    
    # Extract from organized format
    if "**🔹" in llm_output:
        lines = llm_output.split("\n")
        current_person = ""
        
        for line in lines:
            if "**🔹" in line:
                current_person = line.replace("**🔹", "").replace(":**", "").strip()
            elif line.strip().startswith("•") and current_person:
                task = line.strip("• ").strip()
                if task and len(task) > 3:
                    action_tasks.append({
                        "task": task, 
                        "assigned_to": current_person, 
                        "due_date": "Not specified"
                    })
    
    print(f"✅ Extracted {len(action_tasks)} organized action tasks")
    return {"action_tasks": action_tasks}