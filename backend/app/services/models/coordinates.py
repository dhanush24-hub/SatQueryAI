from typing import List, Optional, Tuple
import rasterio.warp
from app.core.logging import logger


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def model_box_to_pixel_box(
    box_norm: List[float],
    image_width: int,
    image_height: int
) -> List[int]:
    """
    Convert model output normalized coordinates [xmin, ymin, xmax, ymax] in [0, 1]
    to integer pixel coordinates in the original raster dimensions [xmin, ymin, xmax, ymax].
    Clamps bounds safely to prevent out-of-bounds indexing.
    """
    x1, y1, x2, y2 = box_norm
    x1_clamped = clamp(x1, 0.0, 1.0)
    y1_clamped = clamp(y1, 0.0, 1.0)
    x2_clamped = clamp(x2, 0.0, 1.0)
    y2_clamped = clamp(y2, 0.0, 1.0)

    px_x1 = int(round(min(x1_clamped, x2_clamped) * image_width))
    px_y1 = int(round(min(y1_clamped, y2_clamped) * image_height))
    px_x2 = int(round(max(x1_clamped, x2_clamped) * image_width))
    px_y2 = int(round(max(y1_clamped, y2_clamped) * image_height))

    # Ensure minimum 1px box
    px_x2 = max(px_x2, px_x1 + 1)
    px_y2 = max(px_y2, px_y1 + 1)

    return [px_x1, px_y1, min(px_x2, image_width), min(px_y2, image_height)]


def pixel_box_to_canvas_box(
    pixel_box: List[int],
    image_width: int,
    image_height: int
) -> List[float]:
    """
    Convert integer pixel box [xmin, ymin, xmax, ymax] into normalized
    canvas percentages [x_min_pct, y_min_pct, width_pct, height_pct] (0.0 - 100.0%).
    """
    px_x1, px_y1, px_x2, px_y2 = pixel_box
    x_pct = round((px_x1 / image_width) * 100.0, 2)
    y_pct = round((px_y1 / image_height) * 100.0, 2)
    w_pct = round(((px_x2 - px_x1) / image_width) * 100.0, 2)
    h_pct = round(((px_y2 - px_y1) / image_height) * 100.0, 2)
    return [x_pct, y_pct, w_pct, h_pct]


def pixel_box_to_geographic_bbox(
    pixel_box: List[int],
    affine_transform: Optional[List[float]],
    crs: Optional[str]
) -> Optional[List[float]]:
    """
    Convert integer pixel box [xmin, ymin, xmax, ymax] into WGS84 geographic
    coordinates [minLon, minLat, maxLon, maxLat] using the raster's affine geotransform.
    Returns None if the raster is not georeferenced.
    """
    if not affine_transform or len(affine_transform) < 6 or not crs:
        return None

    try:
        a, b, c, d, e, f = affine_transform[:6]
        px_x1, px_y1, px_x2, px_y2 = pixel_box

        # Project 4 corners
        corners = [
            (px_x1, px_y1),
            (px_x2, px_y1),
            (px_x2, px_y2),
            (px_x1, px_y2)
        ]
        proj_xs = [a * col + b * row + c for col, row in corners]
        proj_ys = [d * col + e * row + f for col, row in corners]

        if crs.upper() in ("EPSG:4326", "WGS84", "OGC:CRS84"):
            lons = proj_xs
            lats = proj_ys
        else:
            lons, lats = rasterio.warp.transform(crs, "EPSG:4326", proj_xs, proj_ys)

        return [
            round(min(lons), 6),
            round(min(lats), 6),
            round(max(lons), 6),
            round(max(lats), 6)
        ]
    except Exception as err:
        logger.warning(f"Could not calculate geographic bbox from pixel coordinates: {err}")
        return None
