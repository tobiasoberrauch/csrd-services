from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SUPPORTED_FORMATS: list[str] = ["pdf", "docx", "txt"]
    MAX_FILE_SIZE_MB: int = 10
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:8080", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
