from typing import Optional
from ..config.settings import Config

def generate_response(prompt: str, max_retries: int = 3, think: bool = False, max_tokens: int = None) -> Optional[str]:
    """LLM_PROVIDER に応じて LLM バックエンドを切り替えるルーター。
    .env の LLM_PROVIDER: 'gemini' または 'kawarasaki'（デフォルト）
    """
    provider = (Config.LLM_PROVIDER or "kawarasaki").lower()
    if provider == "gemini":
        from ..llm.gemini import generate_response as _gen
    else:
        from ..llm.kawarasaki import generate_response as _gen
    return _gen(prompt, max_retries=max_retries, think=think, max_tokens=max_tokens)
