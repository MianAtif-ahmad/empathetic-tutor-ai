#backend/app/core/config.py 
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./empathetic_tutor.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"
    
    # ML Settings
    MODEL_DIR: str = "./models"
    DATA_DIR: str = "./data"
    
    # Learning rates
    ALPHA_STUDENT: float = 0.1
    ALPHA_GLOBAL: float = 0.01
    ALPHA_EMPATHY: float = 0.05
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()
