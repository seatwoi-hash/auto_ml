from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
from pathlib import Path
import uuid
import os
from dotenv import load_dotenv

from nextcloud_service import NextcloudService


app = FastAPI()
load_dotenv()

nextcloud_service = NextcloudService(
    url=os.getenv("NEXTCLOUD_URL"),
    username=os.getenv("NEXTCLOUD_USER"),
    password=os.getenv("NEXTCLOUD_PASSWORD")
)



# Настройка CORS для разрешения запросов с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



@app.get("/", response_class=HTMLResponse)
async def get_upload_form():
    """Возвращает HTML страницу с формой загрузки"""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/upload/single/")
async def upload_single_file(file: UploadFile = File(...)):
    """Загрузка одного файла"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Можно загружать только изображения")

    # Генерируем уникальное имя
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename

    # Сохраняем файл
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Файл успешно загружен",
        "filename": unique_filename,
        "original_name": file.filename,
        "size": file.size
    }


@app.post("/upload/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Загрузка нескольких файлов одновременно"""
    uploaded_files = []
    errors = []

    for file in files:
        if not file.content_type.startswith("image/"):
            errors.append(f"{file.filename}: не изображение")
            continue

        # Проверка размера (10MB)
        if file.size > 10 * 1024 * 1024:
            errors.append(f"{file.filename}: превышает 10MB")
            continue

        # Генерируем уникальное имя
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename

        print(unique_filename)
        # Сохраняем файл
        try:
            file_bytes = await file.read()
            result = await nextcloud_service.upload_photo(
                file_bytes=file_bytes,
                filename=unique_filename,
                remote_folder="/Photos"
            )

        except Exception as e:
            errors.append(f"{file.filename}: ошибка сохранения - {str(e)}")

    return {
        "message": f"Загружено {len(uploaded_files)} из {len(files)} файлов",
        "files": uploaded_files,
        "errors": errors,
        "total": len(files),
        "success": len(uploaded_files)
    }


@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """Просмотр загруженного файла"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(file_path)


@app.get("/files/")
async def list_uploaded_files():
    """Получить список всех загруженных файлов"""
    files = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "url": f"/uploads/{file_path.name}",
                "created": file_path.stat().st_ctime
            })
    return {"files": files, "count": len(files)}


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Удалить загруженный файл"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Файл не найден")

    file_path.unlink()
    return {"message": f"Файл {filename} удален"}


@app.delete("/files/")
async def delete_all_files():
    """Удалить все загруженные файлы"""
    deleted = 0
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            file_path.unlink()
            deleted += 1
    return {"message": f"Удалено {deleted} файлов"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=80, reload=True)