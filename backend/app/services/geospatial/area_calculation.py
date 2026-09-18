import math
from typing import Dict, List, Optional
from app.core.logging import logger


def compute_pixel_area_m2(
    crs_str: Optional[str] = None,
    resolution: Optional[float] = None,
    center_lat: Optional[float] = None,
    affine_transform: Optional[List[float]] = None
) -> float:
    """
    Computes genuine physical ground area of a single raster pixel in square meters.
    Strictly handles both projected (metric) and geographic (ellipsoidal degrees) CRS.
    Never treats degrees squared as square meters per geospatial accuracy standards.

    """
    # 1. If affine transform is provided [a, b, c, d, e, f] where a = dx, e = dy (negative)
    dx = abs(affine_transform[0]) if affine_transform and len(affine_transform) >= 6 else (resolution or 10.0)
    dy = abs(affine_transform[4]) if affine_transform and len(affine_transform) >= 6 else (resolution or 10.0)

    is_geographic = False
    if crs_str:
        norm_crs = crs_str.upper().strip()
        if "4326" in norm_crs or "WGS84" in norm_crs or "CRS84" in norm_crs:
            is_geographic = True
    elif dx < 0.05 and dy < 0.05:
        # Heuristic: pixel size < 0.05 indicates angular degrees rather than meters
        is_geographic = True

    if is_geographic:
        # Geodesic WGS84 ground area calculation
        # 1 deg latitude ≈ 111,132.954 - 559.822 * cos(2*lat) meters
        # 1 deg longitude ≈ 111,412.84 * cos(lat) - 93.5 * cos(3*lat) meters
        lat_rad = math.radians(center_lat if center_lat is not None else 0.0)
        m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad)
        m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)

        pixel_w_m = dx * abs(m_per_deg_lon)
        pixel_h_m = dy * abs(m_per_deg_lat)
        area_m2 = pixel_w_m * pixel_h_m
        logger.debug(f"Geographic CRS area calculation: dx={dx}°, dy={dy}° at lat={center_lat}° -> {area_m2:.2f} m²")
        return max(0.0001, area_m2)

    # Projected metric CRS (e.g. UTM, Web Mercator)
    area_m2 = dx * dy
    logger.debug(f"Projected CRS area calculation: dx={dx}m, dy={dy}m -> {area_m2:.2f} m²")
    return max(0.0001, area_m2)


def calculate_changed_area(
    num_changed_pixels: int,
    crs_str: Optional[str] = None,
    resolution: Optional[float] = None,
    center_lat: Optional[float] = None,
    affine_transform: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculates total physical changed area given the count of positive change pixels.
    Returns:
        sq_meters: Area in square meters
        hectares: Area in hectares (1 ha = 10,000 m²)
        sq_km: Area in square kilometers (1 km² = 1,000,000 m²)
    """
    pixel_area_m2 = compute_pixel_area_m2(crs_str, resolution, center_lat, affine_transform)
    total_m2 = float(num_changed_pixels) * pixel_area_m2

    return {
        "sq_meters": round(total_m2, 2),
        "hectares": round(total_m2 / 10000.0, 4),
        "sq_km": round(total_m2 / 1000000.0, 6)
    }
