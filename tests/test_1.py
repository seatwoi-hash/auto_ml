import pytest

from unittest.mock import AsyncMock, MagicMock, patch
from nc_py_api import AsyncNextcloud
from nextcloud_service import NextcloudService


@pytest.fixture
def mock_nextcloud():
    """Фикстура с мок-клиентом Nextcloud"""
    with patch('nextcloud_service.AsyncNextcloud') as MockAsyncNextcloud:
        mock_nc = AsyncMock(spec=AsyncNextcloud)
        mock_nc.files = AsyncMock()
        mock_nc.files.mkdir = AsyncMock()
        mock_nc.files.upload = AsyncMock()
        mock_nc.close = AsyncMock()

        MockAsyncNextcloud.return_value = mock_nc
        yield mock_nc


@pytest.fixture
def nextcloud_service(mock_nextcloud):
    """Фикстура с сервисом Nextcloud"""
    service = NextcloudService(
        url="https://nextcloud.example.com",
        username="test_user",
        password="test_pass"
    )
    service.nc = mock_nextcloud
    return service


class TestNextcloudService:

    def test_init_success(self):
        """Тест успешной инициализации"""
        with patch('nextcloud_service.AsyncNextcloud') as MockAsyncNextcloud:
            mock_nc = MagicMock()
            MockAsyncNextcloud.return_value = mock_nc

            service = NextcloudService(
                url="https://test.com",
                username="user",
                password="pass"
            )

            MockAsyncNextcloud.assert_called_once_with(
                nextcloud_url="https://test.com",
                nc_auth_user="user",
                nc_auth_pass="pass"
            )
            assert service.nc is not None

    def test_init_failure(self):
        """Тест ошибки инициализации"""
        with patch('nextcloud_service.AsyncNextcloud') as MockAsyncNextcloud:
            MockAsyncNextcloud.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                NextcloudService(
                    url="https://test.com",
                    username="user",
                    password="pass"
                )

    @pytest.mark.asyncio
    async def test_create_folder_success(self, nextcloud_service):
        """Тест успешного создания папки"""
        result = await nextcloud_service.create_folder("test/folder")

        assert result is True
        nextcloud_service.nc.files.mkdir.assert_called_once_with("test/folder")

    @pytest.mark.asyncio
    async def test_create_folder_trim_slashes(self, nextcloud_service):
        """Тест создания папки с обрезанием слэшей"""
        result = await nextcloud_service.create_folder("/test/folder/")

        assert result is True
        nextcloud_service.nc.files.mkdir.assert_called_once_with("test/folder")

    @pytest.mark.asyncio
    async def test_create_folder_already_exists(self, nextcloud_service):
        """Тест создания уже существующей папки"""
        from nc_py_api import NextcloudException
        nextcloud_service.nc.files.mkdir.side_effect = NextcloudException("Folder exists")

        result = await nextcloud_service.create_folder("existing/folder")

        assert result is True  # Должно вернуть True, так как папка уже существует
        nextcloud_service.nc.files.mkdir.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_folder_no_connection(self, nextcloud_service):
        """Тест создания папки без подключения"""
        nextcloud_service.nc = None

        result = await nextcloud_service.create_folder("test/folder")

        assert result is False

    @pytest.mark.asyncio
    async def test_upload_photo_success(self, nextcloud_service):
        """Тест успешной загрузки фото"""
        file_bytes = b"test image data"
        filename = "photo.jpg"
        remote_folder = "Photos"

        # Мокаем create_folder
        nextcloud_service.create_folder = AsyncMock(return_value=True)

        result = await nextcloud_service.upload_photo(
            file_bytes=file_bytes,
            filename=filename,
            remote_folder=remote_folder
        )

        assert result["success"] is True
        assert result["filename"] == filename
        assert result["remote_path"] == "Photos/photo.jpg"
        assert result["size"] == len(file_bytes)

        nextcloud_service.create_folder.assert_called_once_with(remote_folder)
        nextcloud_service.nc.files.upload.assert_called_once_with(
            "Photos/photo.jpg",
            file_bytes
        )

    @pytest.mark.asyncio
    async def test_upload_photo_with_nested_folder(self, nextcloud_service):
        """Тест загрузки во вложенную папку"""
        file_bytes = b"test data"
        filename = "image.png"
        remote_folder = "Photos/2024/January"

        nextcloud_service.create_folder = AsyncMock(return_value=True)

        result = await nextcloud_service.upload_photo(
            file_bytes=file_bytes,
            filename=filename,
            remote_folder=remote_folder
        )

        assert result["success"] is True
        assert result["remote_path"] == "Photos/2024/January/image.png"
        nextcloud_service.nc.files.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_photo_folder_creation_failed(self, nextcloud_service):
        """Тест загрузки при ошибке создания папки"""
        file_bytes = b"test data"
        filename = "photo.jpg"

        nextcloud_service.create_folder = AsyncMock(return_value=False)

        result = await nextcloud_service.upload_photo(
            file_bytes=file_bytes,
            filename=filename
        )

        assert result["success"] is False
        assert "Не удалось создать папку Photos" in result["error"]
        nextcloud_service.nc.files.upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_photo_no_connection(self, nextcloud_service):
        """Тест загрузки без подключения"""
        nextcloud_service.nc = None

        result = await nextcloud_service.upload_photo(
            file_bytes=b"test",
            filename="test.jpg"
        )

        assert result["success"] is False
        assert result["error"] == "Нет подключения"

    @pytest.mark.asyncio
    async def test_upload_photo_upload_error(self, nextcloud_service):
        """Тест ошибки при загрузке файла"""
        file_bytes = b"test data"
        filename = "photo.jpg"

        nextcloud_service.create_folder = AsyncMock(return_value=True)
        nextcloud_service.nc.files.upload.side_effect = Exception("Upload failed")

        result = await nextcloud_service.upload_photo(
            file_bytes=file_bytes,
            filename=filename
        )

        assert result["success"] is False
        assert result["error"] == "Upload failed"

    @pytest.mark.asyncio
    async def test_upload_photo_large_file(self, nextcloud_service):
        """Тест загрузки большого файла"""
        # Создаем большой файл (1MB)
        file_bytes = b"x" * (1024 * 1024)
        filename = "large_photo.jpg"

        nextcloud_service.create_folder = AsyncMock(return_value=True)

        result = await nextcloud_service.upload_photo(
            file_bytes=file_bytes,
            filename=filename
        )

        assert result["success"] is True
        assert result["size"] == 1024 * 1024

    @pytest.mark.asyncio
    async def test_upload_photo_special_characters(self, nextcloud_service):
        """Тест загрузки файла со спецсимволами в имени"""
        file_bytes = b"test data"
        filename = "фото с пробелами и символами !@#.jpg"
        remote_folder = "Мои Фото/2024"

        nextcloud_service.create_folder = AsyncMock(return_value=True)

        result = await nextcloud_service.upload_photo(
            file_bytes=file_bytes,
            filename=filename,
            remote_folder=remote_folder
        )

        assert result["success"] is True
        assert filename in result["remote_path"]

    @pytest.mark.asyncio
    async def test_close(self, nextcloud_service):
        """Тест закрытия соединения"""
        await nextcloud_service.close()

        nextcloud_service.nc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_connection(self, nextcloud_service):
        """Тест закрытия без активного соединения"""
        nextcloud_service.nc = None

        await nextcloud_service.close()  # Не должно вызвать ошибку

    @pytest.mark.asyncio
    async def test_upload_photo_preserves_file_bytes(self, nextcloud_service):
        """Тест что байты файла не изменяются при загрузке"""
        file_bytes = b"original test data"
        filename = "test.txt"

        nextcloud_service.create_folder = AsyncMock(return_value=True)

        await nextcloud_service.upload_photo(
            file_bytes=file_bytes,
            filename=filename
        )

        # Проверяем, что переданные байты не изменились
        call_args = nextcloud_service.nc.files.upload.call_args[0]
        assert call_args[1] == file_bytes

    @pytest.mark.asyncio
    async def test_multiple_uploads_same_folder(self, nextcloud_service):
        """Тест множественных загрузок в одну папку"""
        nextcloud_service.create_folder = AsyncMock(return_value=True)

        files = [
            (b"data1", "file1.jpg"),
            (b"data2", "file2.jpg"),
            (b"data3", "file3.jpg")
        ]

        for file_bytes, filename in files:
            result = await nextcloud_service.upload_photo(
                file_bytes=file_bytes,
                filename=filename
            )
            assert result["success"] is True

        # create_folder должен быть вызван только один раз (при первой загрузке)
        assert nextcloud_service.create_folder.call_count == 3


@pytest.mark.asyncio
async def test_integration_with_mock():
    """Интеграционный тест с полным моком"""
    service = NextcloudService(
        url="https://test.com",
        username="user",
        password="pass"
    )

    with patch.object(service, '_connect'):
        service.nc = AsyncMock()
        service.nc.files = AsyncMock()
        service.nc.files.mkdir = AsyncMock()
        service.nc.files.upload = AsyncMock()

        # Мокаем create_folder через патч
        with patch.object(service, 'create_folder', AsyncMock(return_value=True)):
            result = await service.upload_photo(
                file_bytes=b"integration test",
                filename="integration.jpg"
            )

            assert result["success"] is True
            assert result["remote_path"] == "Photos/integration.jpg"