from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SUPPORTED_FORMATS: list[str] = ["pdf", "docx", "txt"]
    MAX_FILE_SIZE_MB: int = 10
    SERVER_HOST: str = "127.0.0.1"
    SERVER_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()