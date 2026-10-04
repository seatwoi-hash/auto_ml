import io
import sys

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

import nextcloud_service
import service_ml


class FakeClassifier:
    instances = []

    def __init__(self, model_path):
        self.model_path = model_path
        self.calls = []
        self.__class__.instances.append(self)

    async def predict_from_bytes(self, file_bytes, filename=None):
        self.calls.append((file_bytes, filename))
        try:
            Image.open(io.BytesIO(file_bytes)).verify()
        except Exception as error:
            return {"success": False, "error": str(error)}
        return {
            "success": True,
            "class_idx": 3,
            "class_name": "nature",
            "confidence": 0.95,
        }


class FakeNextcloudService:
    instances = []

    def __init__(self, url, username, password, **kwargs):
        self.connection = (url, username, password)
        self.uploads = []
        self.fail_upload = False
        self.__class__.instances.append(self)

    async def upload_photo(self, file_bytes, filename, remote_folder="Photos"):
        self.uploads.append((file_bytes, filename, remote_folder))
        if self.fail_upload:
            return {"success": False, "error": "Nextcloud unavailable"}
        return {
            "success": True,
            "filename": filename,
            "remote_path": f"{remote_folder}/{filename}",
            "size": len(file_bytes),
        }

    async def close(self):
        return None

    async def healthcheck(self):
        return True


def make_png():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="green").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    FakeClassifier.instances.clear()
    FakeNextcloudService.instances.clear()
    monkeypatch.setattr(service_ml, "PhotoClassifier", FakeClassifier)
    monkeypatch.setattr(nextcloud_service, "NextcloudService", FakeNextcloudService)
    monkeypatch.setenv("NEXTCLOUD_URL", "http://nextcloud:80")
    monkeypatch.setenv("NEXTCLOUD_USER", "test-user")
    monkeypatch.setenv("NEXTCLOUD_PASSWORD", "test-password")
    monkeypatch.setenv("API_KEY", "test-api-key-123456")
    monkeypatch.setenv("MODEL_PATH", "model_checkpoint.pth")
    sys.modules.pop("app", None)

    import app

    app.UPLOAD_DIR = tmp_path
    app.app.state.classifier = FakeClassifier("model_checkpoint.pth")
    app.app.state.nextcloud = FakeNextcloudService(
        "http://nextcloud:80", "test-user", "test-password"
    )
    yield app
    sys.modules.pop("app", None)


def upload_file(name, contents, content_type):
    return UploadFile(
        file=io.BytesIO(contents),
        filename=name,
        headers=Headers({"content-type": content_type}),
        size=len(contents),
    )


def test_service_initializes_dependencies_from_environment(app_module):
    assert FakeClassifier.instances[0].model_path == "model_checkpoint.pth"
    assert FakeNextcloudService.instances[0].connection == (
        "http://nextcloud:80",
        "test-user",
        "test-password",
    )


def test_api_key_authentication(app_module):
    app_module.require_api_key("test-api-key-123456")

    with pytest.raises(HTTPException) as error:
        app_module.require_api_key("wrong-api-key-123456")

    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_health_endpoints(app_module):
    assert await app_module.health_live() == {"status": "ok"}
    assert await app_module.health_ready() == {"status": "ready"}


@pytest.mark.asyncio
async def test_root_returns_upload_page(app_module):
    response = await app_module.get_upload_form()

    assert response.status_code == 200
    assert "Загрузите несколько фото" in response.body.decode()


@pytest.mark.asyncio
async def test_single_upload_saves_image_with_generated_name(app_module):
    payload = await app_module.upload_single_file(
        upload_file("photo.png", make_png(), "image/png")
    )

    assert payload["original_name"] == "photo.png"
    assert payload["filename"].endswith(".png")
    assert (app_module.UPLOAD_DIR / payload["filename"]).read_bytes() == make_png()


@pytest.mark.asyncio
async def test_single_upload_rejects_non_image(app_module):
    with pytest.raises(HTTPException) as error:
        await app_module.upload_single_file(
            upload_file("notes.txt", b"hello", "text/plain")
        )

    assert error.value.status_code == 400
    assert error.value.detail == "Можно загружать только изображения"


@pytest.mark.asyncio
async def test_file_list_download_and_delete(app_module):
    stored_file = app_module.UPLOAD_DIR / "stored.jpg"
    stored_file.write_bytes(b"image contents")

    listing = await app_module.list_uploaded_files()
    download = await app_module.get_uploaded_file("stored.jpg")
    deletion = await app_module.delete_file("stored.jpg")

    assert listing["count"] == 1
    assert listing["files"][0]["name"] == "stored.jpg"
    assert download.path == stored_file
    assert deletion["message"] == "Файл stored.jpg удален"
    assert not stored_file.exists()
    with pytest.raises(HTTPException) as error:
        await app_module.get_uploaded_file("stored.jpg")
    assert error.value.status_code == 404


def test_resolve_upload_path_rejects_unsafe_name(app_module):
    with pytest.raises(HTTPException) as error:
        app_module.resolve_upload_path("..")

    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_multiple_upload_classifies_and_sends_file_to_nextcloud(app_module):
    payload = await app_module.upload_multiple_files(
        [upload_file("forest.png", make_png(), "image/png")]
    )

    assert payload["success"] == 1
    assert payload["errors"] == []
    assert payload["files"][0]["class_name"] == "nature"
    assert payload["files"][0]["confidence"] == 0.95

    uploaded_bytes, stored_name, folder = FakeNextcloudService.instances[0].uploads[0]
    assert uploaded_bytes == make_png()
    assert stored_name.endswith(".png")
    assert folder == "nature"


@pytest.mark.asyncio
async def test_multiple_upload_reports_classifier_error(app_module):
    payload = await app_module.upload_multiple_files(
        [upload_file("broken.jpg", b"broken", "image/jpeg")]
    )

    assert payload["success"] == 0
    assert payload["files"] == []
    assert len(payload["errors"]) == 1
    assert payload["errors"][0].startswith("broken.jpg:")
    assert FakeNextcloudService.instances[0].uploads == []


@pytest.mark.asyncio
async def test_multiple_upload_rejects_non_image(app_module):
    payload = await app_module.upload_multiple_files(
        [upload_file("notes.txt", b"hello", "text/plain")]
    )

    assert payload["success"] == 0
    assert payload["files"] == []
    assert payload["errors"] == ["notes.txt: можно загружать только изображения"]
    assert FakeClassifier.instances[0].calls == []


@pytest.mark.asyncio
async def test_multiple_upload_rejects_fake_image_content(app_module):
    payload = await app_module.upload_multiple_files(
        [upload_file("fake.jpg", b"not-an-image", "image/jpeg")]
    )

    assert payload["success"] == 0
    assert payload["errors"] == ["fake.jpg: содержимое не является изображением"]


@pytest.mark.asyncio
async def test_multiple_upload_limits_number_of_files(app_module):
    files = [
        upload_file(f"{index}.png", make_png(), "image/png")
        for index in range(app_module.MAX_FILES_PER_REQUEST + 1)
    ]

    with pytest.raises(HTTPException) as error:
        await app_module.upload_multiple_files(files)

    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_multiple_upload_reports_nextcloud_failure(app_module):
    FakeNextcloudService.instances[0].fail_upload = True

    payload = await app_module.upload_multiple_files(
        [upload_file("forest.png", make_png(), "image/png")]
    )

    assert payload["success"] == 0
    assert payload["files"] == []
    assert payload["errors"] == ["forest.png: Nextcloud unavailable"]


@pytest.mark.asyncio
async def test_read_upload_rejects_file_over_size_limit(app_module):
    oversized = b"x" * (app_module.MAX_FILE_SIZE + 1)

    with pytest.raises(ValueError, match="превышает 10MB"):
        await app_module.read_upload(upload_file("large.jpg", oversized, "image/jpeg"))


@pytest.mark.asyncio
async def test_delete_all_files(app_module):
    (app_module.UPLOAD_DIR / "one.jpg").write_bytes(b"one")
    (app_module.UPLOAD_DIR / "two.jpg").write_bytes(b"two")

    response = await app_module.delete_all_files()

    assert response["message"] == "Удалено 2 файлов"
    assert list(app_module.UPLOAD_DIR.iterdir()) == []
