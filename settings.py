import os
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Settings(BaseModel):
    nextcloud_url: HttpUrl
    nextcloud_user: str = Field(min_length=1)
    nextcloud_password: str = Field(min_length=1)
    api_key: str = Field(min_length=16)
    model_path: Path = Path("model_checkpoint.pth")
    max_file_size: int = Field(default=10 * 1024 * 1024, gt=0)
    max_files_per_request: int = Field(default=20, ge=1, le=100)
    nextcloud_timeout: float = Field(default=15.0, gt=0)
    nextcloud_retry_attempts: int = Field(default=3, ge=1, le=10)
    cors_origins: list[str] = ["http://localhost:8877"]

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        if "*" in origins:
            raise ValueError("CORS_ORIGINS cannot contain '*' in production")
        return origins


def load_settings() -> Settings:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:8877")
    return Settings(
        nextcloud_url=os.getenv("NEXTCLOUD_URL", ""),
        nextcloud_user=os.getenv("NEXTCLOUD_USER", ""),
        nextcloud_password=os.getenv("NEXTCLOUD_PASSWORD", ""),
        api_key=os.getenv("API_KEY", ""),
        model_path=Path(os.getenv("MODEL_PATH", "model_checkpoint.pth")),
        max_file_size=int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024))),
        max_files_per_request=int(os.getenv("MAX_FILES_PER_REQUEST", "20")),
        nextcloud_timeout=float(os.getenv("NEXTCLOUD_TIMEOUT", "15")),
        nextcloud_retry_attempts=int(os.getenv("NEXTCLOUD_RETRY_ATTEMPTS", "3")),
        cors_origins=[
            origin.strip() for origin in origins.split(",") if origin.strip()
        ],
    )
