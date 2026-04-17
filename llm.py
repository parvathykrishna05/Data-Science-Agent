import os
import time
from dotenv import load_dotenv

load_dotenv()

def call_llm(system_prompt: str, user_message: str, retries: int = 3) -> str:
    """
    Universal LLM Wrapper. 
    Prioritizes Groq (fastest), then Gemini, then Anthropic.
    """
    
    # 1. Try Groq (Best Free Tier - Insanely Fast)
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        model=model,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if "429" in str(e) and attempt < retries - 1:
                        print(f"Groq Rate Limit. Retrying in 5s...")
                        time.sleep(5)
                    else:
                        raise e
            # Return after successful completion of retry loop (though it returns inside)
        except ImportError:
            raise Exception("GROQ_API_KEY found but 'groq' is not installed. Please run: pip install groq")
        except Exception as e:
            raise Exception(f"Groq API Error: {str(e)}")

    # 2. Try Google Gemini (Fallback if no Groq)
    elif os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_message}"
            
            models_to_try = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro']
            for model_id in models_to_try:
                try:
                    for attempt in range(retries):
                        try:
                            response = client.models.generate_content(model=model_id, contents=full_prompt)
                            return response.text
                        except Exception as e:
                            if ("429" in str(e) or "503" in str(e)) and attempt < retries - 1:
                                sleep_time = 10 if "429" in str(e) else 2 ** attempt
                                time.sleep(sleep_time)
                            else:
                                raise e
                except Exception as e:
                    if ("404" in str(e) or "limit: 0" in str(e)) and model_id != models_to_try[-1]:
                        continue
                    raise e
        except Exception as e:
            raise Exception(f"Gemini API Error: {str(e)}")

    # 3. Fallback to Anthropic
    elif os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        try:
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            for attempt in range(retries):
                try:
                    response = client.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_message}],
                    )
                    return response.content[0].text
                except Exception as e:
                    if ("503" in str(e) or "529" in str(e)) and attempt < retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise e
        except anthropic.BadRequestError as e:
            if "credit balance is too low" in str(e):
                raise Exception("ANTHROPIC API ERROR: Account out of credits. Please use Groq or Gemini.")
            raise e
    
    else:
        raise Exception("❌ No API key found! Please set GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in your .env file.")
