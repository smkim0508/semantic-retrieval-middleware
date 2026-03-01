# main app settings/configs
# loads in env variables, sets any project default values

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pydantic import Field

# explicitly locate env file
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

class ServiceSettings(BaseSettings):
    """
    The main service settings. pydantic-settings auto-parses fields from .env.
    """

    # default setting with env file path
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # auto parsed fields from .env
    # main db
    MAIN_DB_POOL_SIZE: int = Field(default=5, description="Number of connections to keep in the pool.")
    MAIN_DB_MAX_OVERFLOW: int = Field(default=10, description="Max 'overflow' connections beyond pool_size.")
    MAIN_DB_POOL_TIMEOUT: int = Field(default=30, description="Seconds to wait before giving up on getting a connection.")
    MAIN_DB_POOL_RECYCLE: int = Field(default=1800, description="Recycle connections after this many seconds.")
    
    MAIN_DB_USER: str
    MAIN_DB_PW: str
    MAIN_DB_HOST: str
    MAIN_DB_PORT: str
    MAIN_DB_NAME: str

    # gemini
    GEMINI_API_KEY: str

    # redis
    REDIS_URL: str = Field(default="redis://localhost:6379")

# lru cache to return service setting instance
@lru_cache()
def get_service_settings() -> ServiceSettings:
    return ServiceSettings() # type: ignore
