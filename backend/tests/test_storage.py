import os
import shutil
import tempfile
from app.services.storage.local import LocalStorageService


def test_local_storage_lifecycle():
    temp_dir = tempfile.mkdtemp()
    try:
        storage = LocalStorageService(base_dir=temp_dir)

        filename = "satellite_patch_01.tif"
        content = b"GEOTIFF_DUMMY_HEADER_BYTES_12345"

        # Check doesn't exist
        assert not storage.exists(filename)
        assert storage.get(filename) is None

        # Save
        saved_path = storage.save(content, filename)
        assert os.path.exists(saved_path)
        assert storage.exists(filename)

        # Get
        retrieved = storage.get(filename)
        assert retrieved == content

        # Delete
        assert storage.delete(filename)
        assert not storage.exists(filename)
        assert storage.get(filename) is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_storage_path_traversal_protection():
    temp_dir = tempfile.mkdtemp()
    try:
        storage = LocalStorageService(base_dir=temp_dir)
        content = b"TEST"

        # Path traversal should be stripped to base name
        saved_path = storage.save(content, "../../evil.txt")
        assert os.path.dirname(saved_path) == temp_dir
        assert os.path.basename(saved_path) == "evil.txt"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
