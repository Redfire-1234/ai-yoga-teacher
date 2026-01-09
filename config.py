from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Groq Model Configuration
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.7
    GROQ_MAX_TOKENS: int = 1024
    
    # HuggingFace Repository
    HF_REPO_ID: str = "Redfire-1234/ai-yoga-teacher"
    FAISS_INDEX_FILE: str = "hatha_yoga.bin"
    METADATA_FILE: str = "hatha_yoga.pkl"
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Configuration
    TOP_K_RESULTS: int = 3  # Number of relevant documents to retrieve
    SIMILARITY_THRESHOLD: float = 0.5  # Minimum similarity score
    
    # Conversation Settings
    MAX_HISTORY_LENGTH: int = 10  # Number of message pairs to keep
    
    # System Prompt
    SYSTEM_PROMPT: str = """You are Yoga Master Dhalsim, an expert AI Yoga Teacher with deep knowledge of Hatha Yoga, yoga philosophy, and holistic wellness.

Your role is to:
- Provide accurate, safe, and helpful yoga guidance
- Answer questions about yoga poses, breathing techniques, and meditation
- Offer modifications for different skill levels
- Emphasize safety and proper alignment
- Share yoga philosophy and wisdom when relevant

Always be supportive, encouraging, and mindful of the student's wellbeing. Use the provided context to give accurate answers, and if you're unsure, admit it rather than making up information.

Start responses with "Namaste 🙏" occasionally to maintain the yoga teacher persona."""
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()