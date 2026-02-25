from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    
    API_V1_STR : str = "/api/v1"
    PROJECT_NAME : str = "Bob Movie Recommendation"
    
    DATABASE_URL: str
    SECRET_KEY: str
    DEBUG: bool = True
    
    MODEL_PATH: str = "models/best_ncf_model.pth"
    
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Pydantic v2 방식
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

settings = Settings()