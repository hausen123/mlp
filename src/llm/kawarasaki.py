import time
import requests
from typing import Optional
from ..config.settings import Config

def generate_response(prompt: str, max_retries: int = 3, think: bool = False, max_tokens: int = None) -> Optional[str]:
    """kawarasaki02 Qwen3 API でレスポンスを生成する。"""
    for attempt in range(max_retries):
        try:
            payload = {
                "prompt": prompt,
                "chat_history": [],
                "think": think,
            }
            response = requests.post(Config.LLM_API_URL, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result.get("response", "")
                print(f"API error: {result}")
                return None
            print(f"HTTP {response.status_code}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"Error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None
