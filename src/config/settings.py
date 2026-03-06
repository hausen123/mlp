import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI")
    NEO4J_USER: Optional[str] = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")
    OLLAMA_URL: Optional[str] = os.getenv("OLLAMA_URL")
    QWEN_MODEL: Optional[str] = os.getenv("QWEN_MODEL")
    API_BASE_URL: Optional[str] = os.getenv("API_BASE_URL")
    LLM_PROVIDER: Optional[str] = os.getenv("LLM_PROVIDER", "kawarasaki")
    LLM_API_URL: Optional[str] = os.getenv("LLM_API_URL", "http://kawarasaki02.info/llm/query/")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: Optional[str] = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")