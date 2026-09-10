from abc import ABC, abstractmethod
from typing import Optional


class StorageService(ABC):
    @abstractmethod
    def save(self, content: bytes, filename: str) -> str:
        """Save raw bytes to storage and return the relative or absolute path/URI."""
        pass

    @abstractmethod
    def get(self, filename: str) -> Optional[bytes]:
        """Retrieve raw bytes for a given stored filename or path."""
        pass

    @abstractmethod
    def exists(self, filename: str) -> bool:
        """Check whether a given filename exists in storage."""
        pass

    @abstractmethod
    def delete(self, filename: str) -> bool:
        """Delete a file from storage. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    def get_full_path(self, filename: str) -> str:
        """Resolve full filesystem path for a stored filename."""
        pass
