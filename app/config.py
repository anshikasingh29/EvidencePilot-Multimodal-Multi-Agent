from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_verifier_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_path: str = "./data/chroma"
    upload_dir: str = "./data/uploads"
    top_k: int = Field(default=5, ge=1, le=12)
    max_file_mb: int = Field(default=15, ge=1, le=50)

@lru_cache
def get_settings() -> Settings:
    return Settings()
