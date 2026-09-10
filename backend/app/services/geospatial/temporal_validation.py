from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from app.domain.modalities import Modality, ImageFormat
from app.schemas.image_asset import ImageAsset
from app.schemas.compatibility import PairType
from app.services.geospatial.compatibility import check_pair_compatibility
from app.core.logging import logger


class TemporalValidationError(ValueError):
    """Exception raised when bi-temporal validation fails."""
    pass


def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Safely parses ISO datetime string if present."""
    if not dt_str:
        return None
    try:
        clean_str = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean_str)
    except Exception:
        return None


def validate_temporal_pair(
    assets: List[ImageAsset],
    user_order: Optional[List[str]] = None
) -> Tuple[ImageAsset, ImageAsset, Dict[str, Any]]:
    """
    Validates a bi-temporal image pair and determines verified T1 and T2 roles.
    
    Args:
        assets: List of ImageAsset objects (must contain exactly 2).
        user_order: Optional explicit list of asset IDs in order [T1_id, T2_id].
        
    Returns:
        (asset_t1, asset_t2, temporal_metadata)
        
    Raises:
        TemporalValidationError if validation conditions fail.
    """
    if len(assets) != 2:
        raise TemporalValidationError(
            f"Bi-temporal change analysis requires exactly 2 images, but received {len(assets)}."
        )

    a1, a2 = assets[0], assets[1]

    # 1. Modality compatibility (optical vs optical, optical vs multispectral; SAR cannot be mixed with optical without fusion)
    if a1.modality == Modality.SAR or a2.modality == Modality.SAR:
        raise TemporalValidationError(
            "Optical-SAR cross-sensor change detection is not supported in the bi-temporal change pipeline. "
            "Both acquisitions must be Optical or Multispectral imagery."
        )

    # 2. Format validation
    valid_formats = [ImageFormat.GEOTIFF, ImageFormat.TIFF, ImageFormat.PNG]
    if a1.format not in valid_formats or a2.format not in valid_formats:
        raise TemporalValidationError(
            f"Unsupported image formats for temporal analysis: {a1.format.value}, {a2.format.value}."
        )

    # 3. Spatial and CRS compatibility check
    compat = check_pair_compatibility(a1, a2, pair_type=PairType.TEMPORAL)
    is_benchmark = (
        a1.validation_status == "benchmark" or
        a2.validation_status == "benchmark" or
        (isinstance(a1.metadata, dict) and a1.metadata.get("benchmark_mode")) or
        (isinstance(a2.metadata, dict) and a2.metadata.get("benchmark_mode"))
    )
    if not is_benchmark and (not compat.has_spatial_overlap or compat.spatial_overlap_percentage <= 0.0):
        raise TemporalValidationError(
            "Bi-temporal images have no geographic spatial intersection (0.0% overlap)."
        )

    if not compat.compatible:
        warn_msg = "; ".join(compat.warnings) if compat.warnings else "Spatial/modal incompatibility"
        raise TemporalValidationError(f"Images are incompatible for temporal analysis: {warn_msg}")

    # 4. Temporal Chronology Determination (T1 = pre-change, T2 = post-change)
    t1_dt = parse_iso_datetime(a1.acquisition_time)
    t2_dt = parse_iso_datetime(a2.acquisition_time)
    order_provenance = "metadata"

    if user_order and len(user_order) == 2:
        # Explicit user-specified ordering
        id_map = {a.id: a for a in assets}
        if user_order[0] in id_map and user_order[1] in id_map:
            t1_asset = id_map[user_order[0]]
            t2_asset = id_map[user_order[1]]
            order_provenance = "user_specified"
            logger.info(f"Using user-specified temporal ordering: T1={t1_asset.id}, T2={t2_asset.id}")
        else:
            raise TemporalValidationError(
                f"User-specified order references unknown asset IDs: {user_order}"
            )
    elif t1_dt and t2_dt:
        if t1_dt < t2_dt:
            t1_asset, t2_asset = a1, a2
        elif t2_dt < t1_dt:
            t1_asset, t2_asset = a2, a1
        else:
            # Identical timestamps
            t1_asset, t2_asset = a1, a2
            order_provenance = "input_order_identical_timestamps"
            logger.warning("Both assets have identical acquisition timestamps; preserving input order.")
    elif t1_dt and not t2_dt:
        t1_asset, t2_asset = a1, a2
        order_provenance = "partial_metadata_default"
        logger.warning(f"Asset '{a2.id}' is missing acquisition timestamp; assuming input order [T1, T2].")
    elif t2_dt and not t1_dt:
        t1_asset, t2_asset = a1, a2
        order_provenance = "partial_metadata_default"
        logger.warning(f"Asset '{a1.id}' is missing acquisition timestamp; assuming input order [T1, T2].")
    else:
        # Neither has acquisition timestamp
        t1_asset, t2_asset = a1, a2
        order_provenance = "unspecified_input_order"
        logger.info("No acquisition dates found in metadata; defaulting to provided input order [T1, T2].")

    # 5. Grid and resolution checks
    res_diff_ratio = None
    if a1.resolution and a2.resolution and a1.resolution > 0 and a2.resolution > 0:
        res_diff_ratio = round(abs(a1.resolution - a2.resolution) / min(a1.resolution, a2.resolution), 4)

    temporal_meta = {
        "t1_id": t1_asset.id,
        "t2_id": t2_asset.id,
        "t1_acquisition_time": t1_asset.acquisition_time,
        "t2_acquisition_time": t2_asset.acquisition_time,
        "order_provenance": order_provenance,
        "spatial_overlap_percentage": compat.spatial_overlap_percentage,
        "crs_t1": t1_asset.crs,
        "crs_t2": t2_asset.crs,
        "resolution_t1": t1_asset.resolution,
        "resolution_t2": t2_asset.resolution,
        "resolution_difference_ratio": res_diff_ratio,
        "needs_reprojection": compat.requires_reprojection,
        "is_grid_aligned": compat.is_coregistered
    }

    return t1_asset, t2_asset, temporal_meta
