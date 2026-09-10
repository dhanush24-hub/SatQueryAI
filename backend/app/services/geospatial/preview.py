import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from app.domain.modalities import Modality
from app.core.config import settings
from app.core.logging import logger

MAX_PREVIEW_SIZE = 1024


def _percentile_stretch(band_data: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """
    Apply a 2%–98% percentile linear contrast stretch to map values to uint8 [0, 255].
    Preserves nodata areas as zero.
    """
    # Mask invalid or nodata pixels
    valid_mask = np.isfinite(band_data)
    if nodata is not None:
        valid_mask = valid_mask & (band_data != nodata)

    if not np.any(valid_mask):
        return np.zeros(band_data.shape, dtype=np.uint8)

    valid_pixels = band_data[valid_mask]
    p2, p98 = np.percentile(valid_pixels, (2, 98))

    if p98 <= p2:
        # Uniform or low contrast band
        out = np.zeros(band_data.shape, dtype=np.uint8)
        out[valid_mask] = 128
        return out

    stretched = np.clip((band_data - p2) / (p98 - p2), 0.0, 1.0) * 255.0
    out = np.zeros(band_data.shape, dtype=np.uint8)
    out[valid_mask] = stretched[valid_mask].astype(np.uint8)
    return out


def generate_raster_preview(
    raster_path: str,
    asset_id: str,
    modality: Modality = Modality.UNKNOWN,
    rgb_bands: Optional[List[int]] = None,
    output_dir: Optional[str] = None
) -> str:
    """
    Generate a web-displayable preview image (PNG) from real raster pixel values.
    This is strictly for visual inspection and UI display; original analytical rasters are preserved untouched.
    """
    out_dir = Path(output_dir or settings.PREVIEW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{asset_id}.png"

    try:
        with rasterio.open(raster_path) as src:
            width = src.width
            height = src.height
            bands = src.count
            nodata = src.nodata

            # Determine downsampling scale factor for large rasters
            max_dim = max(width, height)
            scale = min(1.0, MAX_PREVIEW_SIZE / max_dim)
            out_w = max(1, int(width * scale))
            out_h = max(1, int(height * scale))

            # Case 1: Single-band (Grayscale or SAR)
            if bands == 1:
                data = src.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.bilinear
                ).astype(np.float32)

                stretched = _percentile_stretch(data, nodata=nodata)
                img = Image.fromarray(stretched, mode="L")
                img.save(str(out_path), format="PNG", optimize=True)
                logger.info(f"Generated single-band grayscale preview for {asset_id}")
                return str(out_path)

            # Case 2: Multi-band (Optical / Multispectral / Multi-pol SAR)
            if rgb_bands and len(rgb_bands) == 3 and all(1 <= b <= bands for b in rgb_bands):
                selected_bands = rgb_bands
            elif bands >= 3:
                selected_bands = [1, 2, 3]  # Standard first 3 bands
            else:
                selected_bands = [1]

            if len(selected_bands) == 3:
                channels = []
                for b_idx in selected_bands:
                    b_data = src.read(
                        b_idx,
                        out_shape=(out_h, out_w),
                        resampling=Resampling.bilinear
                    ).astype(np.float32)
                    channels.append(_percentile_stretch(b_data, nodata=nodata))

                rgb_array = np.stack(channels, axis=-1)
                img = Image.fromarray(rgb_array, mode="RGB")
                img.save(str(out_path), format="PNG", optimize=True)
                logger.info(f"Generated 3-band RGB preview for {asset_id}")
                return str(out_path)
            else:
                b_data = src.read(
                    selected_bands[0],
                    out_shape=(out_h, out_w),
                    resampling=Resampling.bilinear
                ).astype(np.float32)
                stretched = _percentile_stretch(b_data, nodata=nodata)
                img = Image.fromarray(stretched, mode="L")
                img.save(str(out_path), format="PNG", optimize=True)
                logger.info(f"Generated single-band preview for {asset_id}")
                return str(out_path)

    except Exception as e:
        logger.error(f"Failed to generate preview for raster {raster_path}: {str(e)}")
        # Fallback empty 1x1 transparent PNG if decoding preview fails
        img = Image.new("RGBA", (256, 256), (30, 35, 45, 255))
        img.save(str(out_path), format="PNG")
        return str(out_path)
