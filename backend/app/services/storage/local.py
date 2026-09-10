import os
import shutil
from pathlib import Path
from typing import Optional
from app.services.storage.base import StorageService
from app.core.config import settings
from app.core.logging import logger


class LocalStorageService(StorageService):
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or settings.STORAGE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LocalStorageService at: {self.base_dir}")

    def _resolve_path(self, filename: str) -> Path:
        # Prevent directory traversal attacks
        clean_name = os.path.basename(filename)
        return self.base_dir / clean_name

    def save(self, content: bytes, filename: str) -> str:
        target_path = self._resolve_path(filename)
        with open(target_path, "wb") as f:
            f.write(content)
        logger.debug(f"Saved {len(content)} bytes to {target_path}")
        return str(target_path)

    def get(self, filename: str) -> Optional[bytes]:
        target_path = self._resolve_path(filename)
        if not target_path.exists() or not target_path.is_file():
            return None
        with open(target_path, "rb") as f:
            return f.read()

    def exists(self, filename: str) -> bool:
        target_path = self._resolve_path(filename)
        return target_path.exists() and target_path.is_file()

    def delete(self, filename: str) -> bool:
        target_path = self._resolve_path(filename)
        if target_path.exists() and target_path.is_file():
            target_path.unlink()
            logger.debug(f"Deleted file {target_path}")
            return True
        return False

    def get_full_path(self, filename: str) -> str:
        return str(self._resolve_path(filename))


# Singleton instance for dependency injection
storage_service = LocalStorageService()
