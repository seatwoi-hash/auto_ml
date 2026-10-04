import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from nc_py_api import AsyncNextcloud, NextcloudException

logger = logging.getLogger(__name__)

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
FOLDER_ALREADY_EXISTS_STATUS = 405


class NextcloudService:
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        *,
        timeout: float = 15.0,
        retry_attempts: int = 3,
    ) -> None:
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.nc = AsyncNextcloud(
            nextcloud_url=url,
            nc_auth_user=username,
            nc_auth_pass=password,
        )

    async def _call_with_retry(
        self,
        operation: Callable[..., Awaitable[Any]],
        *args: Any,
    ) -> Any:
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return await asyncio.wait_for(
                    operation(*args),
                    timeout=self.timeout,
                )
            except TimeoutError:
                retryable = True
                error: Exception = TimeoutError(
                    f"Nextcloud timeout after {self.timeout}s"
                )
            except NextcloudException as nextcloud_error:
                retryable = nextcloud_error.status_code in TRANSIENT_STATUS_CODES
                error = nextcloud_error

            if not retryable or attempt == self.retry_attempts:
                raise error
            delay = 0.5 * 2 ** (attempt - 1)
            logger.warning(
                "Nextcloud operation failed (attempt %d/%d): %s",
                attempt,
                self.retry_attempts,
                error,
            )
            await asyncio.sleep(delay)
        raise RuntimeError("Недостижимое состояние retry")

    async def healthcheck(self) -> bool:
        if self.nc is None:
            return False
        try:
            await self._call_with_retry(self._get_capabilities)
        except (NextcloudException, TimeoutError):
            logger.warning("Nextcloud healthcheck failed", exc_info=True)
            return False
        return True

    async def _get_capabilities(self) -> dict:
        return await self.nc.capabilities

    async def create_folder(self, folder_path: str) -> bool:
        if self.nc is None:
            return False
        normalized_path = folder_path.strip("/")
        if not normalized_path:
            return False

        try:
            await self._call_with_retry(self.nc.files.mkdir, normalized_path)
        except NextcloudException as error:
            if error.status_code == FOLDER_ALREADY_EXISTS_STATUS:
                logger.debug("Folder already exists: %s", normalized_path)
                return True
            logger.warning(
                "Failed to create folder %s: %s",
                normalized_path,
                error,
            )
            return False
        except TimeoutError:
            logger.warning("Timeout creating folder: %s", normalized_path)
            return False

        logger.info("Folder created: %s", normalized_path)
        return True

    async def upload_photo(
        self,
        file_bytes: bytes,
        filename: str,
        remote_folder: str = "Photos",
    ) -> dict:
        if self.nc is None:
            return {"success": False, "error": "Нет подключения"}
        if not await self.create_folder(remote_folder):
            return {
                "success": False,
                "error": f"Не удалось создать папку {remote_folder}",
            }

        remote_path = f"{remote_folder.rstrip('/')}/{filename}"
        try:
            await self._call_with_retry(
                self.nc.files.upload,
                remote_path,
                file_bytes,
            )
        except (NextcloudException, TimeoutError) as error:
            logger.warning("Upload failed: %s", error)
            return {"success": False, "error": str(error)}

        logger.info("Uploaded file to %s", remote_path)
        return {
            "success": True,
            "filename": filename,
            "remote_path": remote_path,
            "size": len(file_bytes),
        }

    async def close(self) -> None:
        if self.nc is not None:
            await self.nc.close()
