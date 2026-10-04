import asyncio
import io
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

from nextcloud_service import NextcloudService
from schemas import (
    BatchUploadResponse,
    FileListResponse,
    HealthResponse,
    MessageResponse,
    SingleUploadResponse,
)
from service_ml import PhotoClassifier
from settings import load_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
settings = load_settings()


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.classifier = PhotoClassifier(model_path=str(settings.model_path))
    application.state.nextcloud = NextcloudService(
        url=str(settings.nextcloud_url),
        username=settings.nextcloud_user,
        password=settings.nextcloud_password,
        timeout=settings.nextcloud_timeout,
        retry_attempts=settings.nextcloud_retry_attempts,
    )
    try:
        yield
    finally:
        await application.state.nextcloud.close()


app = FastAPI(lifespan=lifespan)

MAX_FILE_SIZE = settings.max_file_size
MAX_FILES_PER_REQUEST = settings.max_files_per_request
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Неверный API key")


def get_classifier() -> PhotoClassifier:
    return app.state.classifier


def get_nextcloud_service() -> NextcloudService:
    return app.state.nextcloud


@app.get("/", response_class=HTMLResponse)
async def get_upload_form():
    """Возвращает HTML страницу с формой загрузки"""
    return HTMLResponse(content=Path("index.html").read_text(encoding="utf-8"))


@app.get("/health/live", response_model=HealthResponse)
async def health_live():
    return {"status": "ok"}


@app.get("/health/ready", response_model=HealthResponse)
async def health_ready():
    ready = hasattr(app.state, "classifier") and hasattr(app.state, "nextcloud")
    if not ready or not await app.state.nextcloud.healthcheck():
        raise HTTPException(status_code=503, detail="Сервис не готов")
    return {"status": "ready"}


def validate_image(file: UploadFile) -> None:
    if not (file.content_type or "").startswith("image/"):
        raise ValueError("можно загружать только изображения")


async def read_upload(file: UploadFile) -> bytes:
    validate_image(file)
    contents = await file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise ValueError("превышает 10MB")
    try:
        await asyncio.to_thread(verify_image, contents)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("содержимое не является изображением") from error
    return contents


def verify_image(contents: bytes) -> None:
    with Image.open(io.BytesIO(contents)) as image:
        image.verify()


def resolve_upload_path(filename: str) -> Path:
    if not filename or filename != Path(filename).name or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Некорректное имя файла")
    return UPLOAD_DIR / filename


@app.post(
    "/upload/single/",
    dependencies=[Depends(require_api_key)],
    response_model=SingleUploadResponse,
)
async def upload_single_file(file: UploadFile = File(...)):
    """Загрузка одного файла"""
    try:
        file_bytes = await read_upload(file)
    except ValueError as error:
        raise HTTPException(400, str(error).capitalize()) from error
    finally:
        await file.close()

    original_name = file.filename or "image"
    unique_filename = f"{uuid.uuid4()}{Path(original_name).suffix}"
    await asyncio.to_thread(
        (UPLOAD_DIR / unique_filename).write_bytes,
        file_bytes,
    )

    return {
        "message": "Файл успешно загружен",
        "filename": unique_filename,
        "original_name": file.filename,
        "size": len(file_bytes),
    }


@app.get("/uploads/{filename}", dependencies=[Depends(require_api_key)])
async def get_uploaded_file(filename: str):
    """Просмотр загруженного файла"""
    file_path = resolve_upload_path(filename)
    if not file_path.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(file_path)


@app.get(
    "/files/",
    dependencies=[Depends(require_api_key)],
    response_model=FileListResponse,
)
async def list_uploaded_files():
    """Получить список всех загруженных файлов"""
    files = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            files.append(
                {
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "url": f"/uploads/{file_path.name}",
                    "created": file_path.stat().st_ctime,
                }
            )
    return {"files": files, "count": len(files)}


@app.delete(
    "/files/{filename}",
    dependencies=[Depends(require_api_key)],
    response_model=MessageResponse,
)
async def delete_file(filename: str):
    """Удалить загруженный файл"""
    file_path = resolve_upload_path(filename)
    if not file_path.exists():
        raise HTTPException(404, "Файл не найден")

    file_path.unlink()
    return {"message": f"Файл {filename} удален"}


@app.delete(
    "/files/",
    dependencies=[Depends(require_api_key)],
    response_model=MessageResponse,
)
async def delete_all_files():
    """Удалить все загруженные файлы"""
    deleted = 0
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            file_path.unlink()
            deleted += 1
    return {"message": f"Удалено {deleted} файлов"}


@app.post(
    "/upload/",
    dependencies=[Depends(require_api_key)],
    response_model=BatchUploadResponse,
)
async def upload_multiple_files(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Можно загрузить не более {MAX_FILES_PER_REQUEST} файлов",
        )
    uploaded_files = []
    errors = []

    for file in files:
        filename = file.filename or "image"
        try:
            file_bytes = await read_upload(file)

            file_extension = Path(filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"

            logger.info("Processing: %s", filename)

            prediction = await get_classifier().predict_from_bytes(
                file_bytes=file_bytes, filename=filename
            )

            if not prediction["success"]:
                errors.append(f"{filename}: {prediction.get('error')}")
                continue

            class_name = prediction["class_name"]
            confidence = prediction["confidence"]

            logger.info(f"class={class_name}, confidence={confidence}")

            # upload в Nextcloud
            result = await get_nextcloud_service().upload_photo(
                file_bytes=file_bytes,
                filename=unique_filename,
                remote_folder=class_name,
            )

            if not result.get("success"):
                errors.append(
                    f"{filename}: {result.get('error', 'ошибка загрузки в Nextcloud')}"
                )
                continue

            logger.info(f"Uploaded: {result}")

            uploaded_files.append(
                {
                    "original_name": filename,
                    "stored_name": unique_filename,
                    "class_name": class_name,
                    "confidence": confidence,
                    "success": True,
                }
            )

        except ValueError as error:
            errors.append(f"{filename}: {error}")
        except Exception as error:
            logger.exception("Ошибка загрузки")
            errors.append(f"{filename}: {error}")
        finally:
            await file.close()

    return {
        "message": f"Загружено {len(uploaded_files)} из {len(files)} файлов",
        "files": uploaded_files,
        "errors": errors,
        "total": len(files),
        "success": len(uploaded_files),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8877, reload=True)
