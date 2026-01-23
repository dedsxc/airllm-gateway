from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Server Config
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # AMD Hardware
    hsa_override_gfx_version: str = Field(..., alias="HSA_OVERRIDE_GFX_VERSION")

    # Model Config
    model_id: str = "unsloth/llama-3-70b-Instruct-bnb-4bit"
    model_display_name: str = "AirLLM-Default"
    compression: str = "4bit"
    max_length: int = 2048

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"

settings = Settings()