import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from app.domain.modalities import Modality, ImageFormat
from app.schemas.image_asset import ImageAsset
from app.services.storage.local import storage_service
from app.db.asset_repository import asset_repository
from app.services.geospatial.validation import validate_file_signature, sanitize_filename
from app.services.geospatial.raster_reader import inspect_raster
from app.services.geospatial.preview import generate_raster_preview
from app.core.logging import logger

router = APIRouter(prefix="/api/uploads", tags=["Uploads"])


@router.post("", response_model=List[ImageAsset], status_code=status.HTTP_201_CREATED)
async def upload_images(
    files: List[UploadFile] = File(...),
    benchmark_mode: bool = Form(False),
    modality: Optional[str] = Form(None)
) -> List[ImageAsset]:
    """
    Ingest remote sensing imagery files (GeoTIFF / TIFF; PNG/JPEG restricted to benchmark mode).
    Inspects real raster metadata with Rasterio, generates display preview, and persists to SQLite registry.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 6:
        raise HTTPException(status_code=400, detail="Maximum of 6 images allowed per session.")

    user_modality = None
    if modality:
        try:
            user_modality = Modality(modality.upper())
        except ValueError:
            user_modality = Modality.UNKNOWN

    assets: List[ImageAsset] = []

    for file in files:
        raw_filename = file.filename or "uploaded_raster.tif"
        content = await file.read()

        # 1. Server-side signature validation
        try:
            detected_format = validate_file_signature(
                content=content,
                filename=raw_filename,
                allow_benchmark_rgb=benchmark_mode
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        clean_name = sanitize_filename(raw_filename)
        asset_id = f"img_{uuid.uuid4().hex[:10]}"
        ext = os.path.splitext(clean_name)[1].lower() or ".tif"
        storage_filename = f"{asset_id}{ext}"

        # 2. Persist raw raster
        internal_saved_path = storage_service.save(content, storage_filename)

        # 3. Real Rasterio metadata inspection
        try:
            raster_meta = inspect_raster(internal_saved_path, user_modality=user_modality)
        except ValueError as e:
            # Clean up stored invalid file
            storage_service.delete(storage_filename)
            raise HTTPException(status_code=400, detail=f"Raster validation failed for {clean_name}: {str(e)}")

        # 4. Generate visual display preview
        preview_path = generate_raster_preview(
            raster_path=internal_saved_path,
            asset_id=asset_id,
            modality=raster_meta.modality
        )

        safe_storage_key = f"storage/{storage_filename}"
        preview_url = f"/api/uploads/{asset_id}/preview"

        asset = ImageAsset(
            id=asset_id,
            filename=storage_filename,
            storage_path=safe_storage_key,
            original_filename=clean_name,
            format=detected_format,
            modality=raster_meta.modality,
            width=raster_meta.width,
            height=raster_meta.height,
            bands=raster_meta.bands,
            crs=raster_meta.crs,
            resolution=raster_meta.resolution,
            bbox=raster_meta.bbox,
            geographic_bbox=raster_meta.geographic_bbox,
            nodata=raster_meta.nodata,
            acquisition_time=raster_meta.acquisition_time,
            sensor=raster_meta.sensor,
            affine_transform=raster_meta.affine_transform,
            driver=raster_meta.driver,
            band_descriptions=raster_meta.band_descriptions,
            dtypes=raster_meta.dtypes,
            preview_url=preview_url,
            has_preview=bool(preview_path),
            metadata={"raw_tags": raster_meta.tags, "file_size_bytes": len(content), "benchmark_mode": benchmark_mode},
            validation_status="valid",
            warnings=raster_meta.warnings
        )

        # 5. Persist to SQLite Asset Registry
        asset_repository.save_asset(
            asset=asset,
            internal_storage_path=internal_saved_path,
            internal_preview_path=preview_path
        )

        assets.append(asset)
        logger.info(f"Successfully processed and cataloged ImageAsset {asset.id} ({asset.original_filename})")

    return assets


@router.get("/previews/{filename}")
async def get_preview_file(filename: str) -> FileResponse:
    """
    Retrieve generated preview image (e.g. change mask PNG) by filename.
    """
    full_path = storage_service.get_full_path(filename)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Preview file not found.")
    return FileResponse(
        path=full_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"}
    )


@router.get("/{asset_id}", response_model=ImageAsset)
async def get_image_asset(asset_id: str) -> ImageAsset:
    """
    Retrieve normalized ImageAsset metadata.
    Does NOT leak absolute server filesystem paths.
    """
    asset = asset_repository.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"ImageAsset '{asset_id}' not found.")
    return asset


@router.get("/{asset_id}/preview")
async def get_image_preview(asset_id: str) -> FileResponse:
    """
    Retrieve browser-displayable preview image (PNG) for an ImageAsset.
    """
    paths = asset_repository.get_internal_paths(asset_id)
    if not paths or not paths.get("preview_path"):
        raise HTTPException(status_code=404, detail=f"Preview for ImageAsset '{asset_id}' not found.")

    preview_path = paths["preview_path"]
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview file is missing from storage.")

    return FileResponse(
        path=preview_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"}
    )
