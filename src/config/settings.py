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
    E5_EMBEDDING_URL: Optional[str] = os.getenv("E5_EMBEDDING_URL")