import os
import re
from typing import Tuple
from app.domain.modalities import ImageFormat

TIFF_SIGNATURES = [
    b"\x49\x49\x2a\x00",  # Little-endian TIFF (II*)
    b"\x4d\x4d\x00\x2a",  # Big-endian TIFF (MM*)
    b"\x49\x49\x2b\x00",  # Little-endian BigTIFF (II+)
    b"\x4d\x4d\x00\x2b",  # Big-endian BigTIFF (MM+)
]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
WEBP_SIGNATURE = b"RIFF"

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_RASTER_DIMENSION = 15000  # Prevent decompression bombs
MAX_BAND_COUNT = 64


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal and special character exploits."""
    base = os.path.basename(filename)
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return clean or "unnamed_raster.tif"


def validate_file_signature(content: bytes, filename: str, allow_benchmark_rgb: bool = False) -> ImageFormat:
    """
    Validate magic bytes and enforce GeoTIFF-first rules.
    PNG/JPEG are only accepted when benchmark_mode is explicitly enabled.
    """
    if len(content) < 8:
        raise ValueError("File is empty or too small to contain valid raster headers.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.")

    header = content[:8]

    # 1. GeoTIFF / TIFF
    if any(header.startswith(sig) for sig in TIFF_SIGNATURES):
        return ImageFormat.GEOTIFF

    # 2. Benchmark PNG
    if header.startswith(PNG_SIGNATURE):
        if not allow_benchmark_rgb:
            raise ValueError(
                "PNG format is restricted to approved benchmark evaluation mode. "
                "Real remote sensing workflows require georeferenced GeoTIFF (.tif / .tiff) rasters."
            )
        return ImageFormat.PNG

    # 3. Benchmark JPEG
    if header.startswith(JPEG_SIGNATURE):
        if not allow_benchmark_rgb:
            raise ValueError(
                "JPEG format is restricted to approved benchmark evaluation mode. "
                "Real remote sensing workflows require georeferenced GeoTIFF (.tif / .tiff) rasters."
            )
        return ImageFormat.JPEG

    # 4. WebP
    if header.startswith(WEBP_SIGNATURE) and b"WEBP" in content[:16]:
        if not allow_benchmark_rgb:
            raise ValueError("WebP format is restricted to benchmark evaluation mode.")
        return ImageFormat.WEBP

    raise ValueError(
        f"Unsupported or invalid raster file signature for '{filename}'. "
        "File must be a valid GeoTIFF raster."
    )
