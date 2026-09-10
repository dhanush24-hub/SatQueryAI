import os
import uuid
from typing import Optional, Tuple
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from app.schemas.image_asset import ImageAsset
from app.domain.modalities import ImageFormat, Modality
from app.services.storage.local import storage_service
from app.db.asset_repository import asset_repository
from app.services.geospatial.raster_reader import inspect_raster
from app.services.geospatial.preview import generate_raster_preview
from app.core.logging import logger

RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "lanczos": Resampling.lanczos,
}


def align_and_reproject_raster(
    source_asset_id: str,
    reference_asset_id: str,
    resampling_name: str = "bilinear"
) -> ImageAsset:
    """
    Reproject and resample source_asset to match the CRS, bounds, and pixel grid of reference_asset.
    Original analytical rasters are preserved untouched.
    Derived aligned raster is saved as a new ImageAsset with explicit provenance.
    """
    source_asset = asset_repository.get_asset(source_asset_id)
    ref_asset = asset_repository.get_asset(reference_asset_id)

    if not source_asset or not ref_asset:
        raise ValueError("Source or reference asset not found in database.")

    source_paths = asset_repository.get_internal_paths(source_asset_id)
    ref_paths = asset_repository.get_internal_paths(reference_asset_id)

    if not source_paths or not ref_paths:
        raise ValueError("Physical storage paths for assets could not be resolved.")

    source_file = source_paths["storage_path"]
    ref_file = ref_paths["storage_path"]

    resampling = RESAMPLING_MAP.get(resampling_name.lower(), Resampling.bilinear)

    with rasterio.open(ref_file) as ref_src:
        dst_crs = ref_src.crs
        dst_transform = ref_src.transform
        dst_width = ref_src.width
        dst_height = ref_src.height

        if not dst_crs:
            raise ValueError("Reference raster does not have a Coordinate Reference System (CRS).")

    with rasterio.open(source_file) as src:
        src_crs = src.crs
        if not src_crs:
            raise ValueError("Source raster does not have a Coordinate Reference System (CRS).")

        band_count = src.count
        src_nodata = src.nodata
        dtype = src.dtypes[0]

        # Allocate output array matching reference raster dimensions
        dst_data = np.zeros((band_count, dst_height, dst_width), dtype=dtype)

        # Reproject each band
        for b_idx in range(1, band_count + 1):
            reproject(
                source=rasterio.band(src, b_idx),
                destination=dst_data[b_idx - 1],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=src_nodata,
                dst_nodata=src_nodata,
            )

        dst_profile = src.profile.copy()
        dst_profile.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": src_nodata,
        })

    # Write new aligned GeoTIFF to storage
    aligned_asset_id = f"img_aligned_{uuid.uuid4().hex[:10]}"
    aligned_filename = f"{aligned_asset_id}.tif"
    temp_target_path = os.path.join(storage_service.base_dir, aligned_filename)

    with rasterio.open(temp_target_path, "w", **dst_profile) as dst:
        dst.write(dst_data)

    logger.info(f"Generated aligned raster {aligned_asset_id} at {temp_target_path}")

    # Inspect the newly generated aligned raster
    raster_meta = inspect_raster(temp_target_path, user_modality=source_asset.modality)

    # Generate preview
    preview_path = generate_raster_preview(
        raster_path=temp_target_path,
        asset_id=aligned_asset_id,
        modality=raster_meta.modality
    )

    warnings = list(raster_meta.warnings)
    warnings.append(
        f"Derived raster aligned from {source_asset_id} to reference {reference_asset_id} using {resampling_name} resampling. "
        "Geometric co-registration is resampled but field-level subpixel alignment should still be verified."
    )

    aligned_asset = ImageAsset(
        id=aligned_asset_id,
        filename=aligned_filename,
        storage_path=f"storage/{aligned_filename}",
        original_filename=f"aligned_{source_asset.original_filename}",
        format=ImageFormat.GEOTIFF,
        modality=source_asset.modality,
        width=raster_meta.width,
        height=raster_meta.height,
        bands=raster_meta.bands,
        crs=raster_meta.crs,
        resolution=raster_meta.resolution,
        bbox=raster_meta.bbox,
        geographic_bbox=raster_meta.geographic_bbox,
        nodata=raster_meta.nodata,
        acquisition_time=source_asset.acquisition_time,
        sensor=source_asset.sensor,
        affine_transform=raster_meta.affine_transform,
        driver=raster_meta.driver,
        band_descriptions=raster_meta.band_descriptions,
        dtypes=raster_meta.dtypes,
        preview_url=f"/api/uploads/{aligned_asset_id}/preview",
        has_preview=True,
        metadata={
            "provenance": {
                "source_asset_id": source_asset_id,
                "reference_asset_id": reference_asset_id,
                "resampling": resampling_name,
                "target_crs": str(dst_crs),
            },
            "raw_tags": raster_meta.tags
        },
        validation_status="valid",
        warnings=warnings
    )

    asset_repository.save_asset(
        asset=aligned_asset,
        internal_storage_path=temp_target_path,
        internal_preview_path=preview_path
    )

    return aligned_asset
