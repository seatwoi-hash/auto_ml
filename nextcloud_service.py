from nc_py_api import AsyncNextcloud
from urllib.parse import quote
import tempfile
from pathlib import Path
import os


class NextcloudService:

    def __init__(self, url: str, username: str, password: str):
        self.url = url
        self.username = username
        self.password = password
        self.nc = None
        self._connect()

    def _connect(self):
        if self.nc is None:
            try:
                self.nc = AsyncNextcloud(
                    nextcloud_url=self.url,
                    nc_auth_user=self.username,
                    nc_auth_pass=self.password
                )
            except Exception as e:
                print(f"Ошибка инициализации клиента Nextcloud: {e}")
                raise e

    def create_folder(self, folder_path: str) -> bool:

        if not self.nc:
            return False

        try:
            if self.nc.files.is_exists(folder_path):
                return True

            self.nc.files.mkdir(folder_path)
            return True

        except Exception as e:
            return False


    async def upload_photo(self, file_bytes: bytes, filename: str, remote_folder: str = "/Photos") -> dict:

        if not self.nc:
            return {"success": False, "error": "Нет подключения"}

        temp_path = None

        try:
            file_extension = Path(filename).suffix
            print(type(file_bytes))
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file_bytes)
                temp_path = tmp_file.name
            print(remote_folder)

            path = f"{remote_folder.rstrip('/')}/{filename}"
            # Синхронная загрузка (но в асинхронной функции можно)
            await self.nc.files.upload(
                path,
                file_bytes
            )

            return {
                "success": True,
                "filename": filename,
                "remote_path": remote_folder,
                "size": len(file_bytes)
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def close(self):
            if self.nc:
                await self.nc.close()
