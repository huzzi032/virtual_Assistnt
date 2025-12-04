# server/tools/llm_tool.py
import aiohttp, os, asyncio
from dotenv import load_dotenv
# from mcp.server import Tool
from mcp.types import TextContent

load_dotenv()

async def call_openai_llm(text: str, prompt: str) -> str:
    """Utility function to call OpenAI GPT-4.1-mini with custom prompt"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL environment variables are required")
    
    # Configure timeout for long content processing (5 minutes total, 30 seconds for connection)
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                base_url,
                headers={"api-key": api_key, "Content-Type": "application/json"},
                json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200}
            ) as resp:
                print(f"🌐 OpenAI API response status: {resp.status}")
                print(f"🌐 OpenAI API response content type: {resp.headers.get('content-type', 'unknown')}")
                
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"❌ OpenAI API error response: {error_text}")
                    raise ValueError(f"OpenAI API returned status {resp.status}: {error_text[:200]}...")
                
                try:
                    result = await resp.json()
                except Exception as json_error:
                    response_text = await resp.text()
                    print(f"❌ Failed to parse JSON response: {json_error}")
                    print(f"❌ Response text: {response_text[:500]}...")
                    raise ValueError(f"Invalid JSON response from OpenAI API: {str(json_error)}")
                
                # Check for API errors
                if "error" in result:
                    raise ValueError(f"OpenAI API Error: {result['error']['message']}")
                
                # Check if choices exist
                if "choices" not in result or not result["choices"]:
                    raise ValueError(f"Unexpected API response: {result}")
                
                output = result["choices"][0]["message"]["content"]
                return output
                
    except asyncio.TimeoutError:
        raise ValueError("LLM request timed out. Please try again.")
    except aiohttp.ClientError as e:
        raise ValueError(f"Network error: {str(e)}")

async def translate_and_process_multilingual(text: str, task_type: str = "general", custom_prompt: str | None = None) -> str:
    """
    Optimized multilingual processing with custom prompts support
    """
    
    # Use custom prompt if provided, otherwise use default based on task type
    if custom_prompt is not None:
        prompt = custom_prompt
    elif task_type == "actions":
        prompt = f"""Read this text and create an action task list from it. Use your intelligence to decide what should be action tasks with assignments.

Text: {text}

Example 1:
Input: "Ahmed, you need to complete the report by Friday. Sarah, please review the budget documents. The presentation is due on Monday."
Output:
Action Tasks:
- task: Complete the report, assigned_to: Ahmed, due_date: Friday
- task: Review the budget documents, assigned_to: Sarah, due_date: Not specified
- task: Complete the presentation, assigned_to: User, due_date: Monday

Example 2:
Input: "Mohammad, your task is to write Python code for the project. Submit it by December 15th, 2025."
Output:
Action Tasks:
- task: Write Python code for the project, assigned_to: Mohammad, due_date: December 15th, 2025

Example 3:
Input: "I'm going shopping later and need to call my friend."
Output:
Action Tasks:
No actionable tasks found

Now process the given text:
Output format:
Action Tasks:
- task: [action], assigned_to: [person], due_date: [date]

If no action tasks, just say:
Action Tasks:
No actionable tasks found"""
    
    elif task_type == "todos":
        prompt = f"""Read this text and create a todo list from it. Use your intelligence to decide what should be todos.

Text: {text}

Example 1:
Input: "I need to buy groceries tomorrow and don't forget to call my mother. Also remind me to pay the electricity bill."
Output:
Summary: Personal reminders about shopping, family call, and bill payment
Todos:
- Buy groceries tomorrow
- Call my mother
- Pay the electricity bill

Example 2:
Input: "Meeting with team at 3pm. I should prepare the presentation and review the quarterly reports."
Output:
Summary: Meeting preparation and document review tasks
Todos:
- Prepare the presentation
- Review the quarterly reports

Now process the given text:
Output format:
Summary: [brief summary]
Todos:
- [todo 1]
- [todo 2]

If no todos, just say:
Summary: [brief summary]
Todos:
No todos found"""

    elif task_type == "calendar":
        prompt = f"""Extract calendar event from: {text}

Output:
Task: [English]
Date: [YYYY-MM-DD] 
Time: [HH:MM]"""

    else:  # general
        prompt = f"""Translate to English: {text}

Output: [English summary]"""

    try:
        result = await call_openai_llm(text, prompt)
        print(f"🌐 Processed: {result[:50]}...")
        return result
    except Exception as e:
        print(f"❌ Translation error: {e}")
        # Raise exception to trigger fallback in calling tools
        raise Exception(f"LLM processing failed: {e}")

# Alias for backward compatibility
async def call_llm(text: str, prompt: str) -> str:
    """Backward compatibility wrapper"""
    return await call_openai_llm(text, prompt)
    async def run(self, args):
        text = args.get("text", "")
        task_type = args.get("task_type", "general")
        
        # Always use optimized multilingual processing
        output = await translate_and_process_multilingual(text, task_type)
        return TextContent(type="text", text=output)
