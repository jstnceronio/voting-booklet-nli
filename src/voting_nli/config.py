
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # CSCS Settings
    api_key: str
    base_url: str
    model: str

settings = Settings()
