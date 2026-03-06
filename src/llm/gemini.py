import time
from google import genai
from google.genai import types
from typing import Optional
from ..config.settings import Config

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client

def generate_response(prompt: str, max_retries: int = 3, think: bool = False, max_tokens: int = None) -> Optional[str]:
    """Gemini API でレスポンスを生成する。"""
    client = _get_client()
    config = types.GenerateContentConfig(
        temperature=0.8,
        top_p=0.9,
        max_output_tokens=max_tokens or 8192,
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=Config.GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            if response.text:
                return response.text
            print("Empty response from Gemini")
            return None
        except Exception as e:
            print(f"Gemini error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None
