import requests
import time
from typing import Optional
from ..config.settings import Config

def generate_response(prompt: str, max_retries: int = 3, think: bool = False, max_tokens: int = None) -> Optional[str]:
    """Ollama APIを使用してレスポンスを生成する"""
    for attempt in range(max_retries):
        try:
            print(f"Generating response from {Config.QWEN_MODEL} (attempt {attempt + 1})")
            payload = {
                "model": Config.QWEN_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": think
            }
            if max_tokens is not None:
                payload["options"] = {"num_predict": max_tokens}
            response = requests.post(f"{Config.OLLAMA_URL}/api/generate", json=payload, timeout=300)
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                print(f"Successfully generated response")
                return response_text
            else:
                print(f"Ollama API request failed with status code {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        except Exception as e:
            print(f"Error generating response (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    print("All retry attempts failed")
    return None
