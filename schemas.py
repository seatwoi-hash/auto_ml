from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class SingleUploadResponse(BaseModel):
    message: str
    filename: str
    original_name: str | None
    size: int


class UploadedFile(BaseModel):
    original_name: str
    stored_name: str
    class_name: str
    confidence: float
    success: bool


class BatchUploadResponse(BaseModel):
    message: str
    files: list[UploadedFile]
    errors: list[str]
    total: int
    success: int


class StoredFile(BaseModel):
    name: str
    size: int
    url: str
    created: float


class FileListResponse(BaseModel):
    files: list[StoredFile]
    count: int


class MessageResponse(BaseModel):
    message: str
