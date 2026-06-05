from nc_py_api import AsyncNextcloud
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NextcloudService:

    def __init__(self, url: str, username: str, password: str):
        self.url = url
        self.username = username
        self.password = password
        self.nc = None
        self._connect()

    def _connect(self):
        logger.info("Connecting to Nextcloud")
        if self.nc is None:
            try:
                self.nc = AsyncNextcloud(
                    nextcloud_url=self.url,
                    nc_auth_user=self.username,
                    nc_auth_pass=self.password
                )
            except Exception as e:
                logger.error(f"Ошибка инициализации клиента Nextcloud: {e}")
                raise e

    async def create_folder(self, folder_path: str) -> bool:
        if not self.nc:
            return False

        try:
            folder_path = folder_path.strip("/")

            logger.info(
                f"create_folder: {folder_path}"
            )

            try:
                await self.nc.files.mkdir(folder_path)
                logger.info(
                    f"folder created: {folder_path}"
                )

            except Exception as e:
                # Папка уже существует —
                # это не ошибка
                logger.info(
                    f"mkdir skipped: {str(e)}"
                )

            return True

        except Exception:
            logger.exception(
                "create_folder failed"
            )
            return False

    async def upload_photo(
            self,
            file_bytes: bytes,
            filename: str,
            remote_folder: str = "Photos"
    ) -> dict:

        if not self.nc:
            return {
                "success": False,
                "error": "Нет подключения"
            }

        try:
            folder_created = await self.create_folder(
                remote_folder
            )

            if not folder_created:
                return {
                    "success": False,
                    "error":
                        f"Не удалось создать "
                        f"папку {remote_folder}"
                }

            path = (
                f"{remote_folder.rstrip('/')}/"
                f"{filename}"
            )

            logger.info(f"upload path: {path}")

            await self.nc.files.upload(
                path,
                file_bytes
            )

            logger.info("upload success")

            return {
                "success": True,
                "filename": filename,
                "remote_path": path,
                "size": len(file_bytes)
            }

        except Exception as e:
            logger.exception(
                "upload failed"
            )

            return {
                "success": False,
                "error": str(e)
            }

    async def close(self):
            if self.nc:
                await self.nc.close()
