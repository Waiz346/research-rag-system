"""Application configuration settings."""
import sys
import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings

if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(sys._MEIPASS)
    BASE_DIR = Path(sys.executable).parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR = BUNDLE_DIR

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"

    # Database configuration (Cloud PostgreSQL / SQLite fallback)
    DATABASE_URL: Optional[str] = None

    # Storage paths
    DATA_DIR: Path = BASE_DIR / "data"
    DOWNLOADS_DIR: Path = BASE_DIR / "data" / "downloads"
    CHROMA_PERSIST_DIRECTORY: str = str(BASE_DIR / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "research_papers"

    # LLM & Generation settings
    LLM_PROVIDER: str = "mock"  # "mock", "openai", "gemini", or "ollama"
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"  # "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
