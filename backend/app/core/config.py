import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_ENV: str = "development"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    PROJECT_NAME: str = "satquery-api"
    VERSION: str = "0.1.0"

    # Storage
    STORAGE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../storage"))
    PREVIEW_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../storage/previews"))

    # Database & Cache
    DATABASE_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../storage/satquery.db"))
    DATABASE_URL: Optional[str] = None
    MODEL_CACHE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models_cache"))

    # Hardware & Model Providers
    HF_TOKEN: Optional[str] = None
    DEVICE: str = "cpu"
    CUDA_VISIBLE_DEVICES: Optional[str] = None

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"
    ]


settings = Settings()
