from fastapi import APIRouter, HTTPException, status
from app.schemas.compatibility import (
    PairCompatibilityRequest,
    PairCompatibilityReport,
    AlignmentRequest,
)
from app.schemas.image_asset import ImageAsset
from app.db.asset_repository import asset_repository
from app.services.geospatial.compatibility import check_pair_compatibility
from app.services.geospatial.alignment import align_and_reproject_raster
from app.core.logging import logger

router = APIRouter(prefix="/api/compatibility", tags=["Compatibility"])


@router.post("/check", response_model=PairCompatibilityReport)
async def evaluate_pair_compatibility(request: PairCompatibilityRequest) -> PairCompatibilityReport:
    """
    Evaluate geographic overlap, resolution match, and modality compatibility between two ImageAssets.
    Does not assume co-registration solely from matching CRS or dimensions.
    """
    asset_a = asset_repository.get_asset(request.asset_id_a)
    if not asset_a:
        raise HTTPException(status_code=404, detail=f"ImageAsset '{request.asset_id_a}' not found.")

    asset_b = asset_repository.get_asset(request.asset_id_b)
    if not asset_b:
        raise HTTPException(status_code=404, detail=f"ImageAsset '{request.asset_id_b}' not found.")

    report = check_pair_compatibility(asset_a, asset_b, pair_type=request.pair_type)
    logger.info(
        f"Compatibility check for {request.asset_id_a} and {request.asset_id_b}: "
        f"overlap={report.spatial_overlap_percentage}%, compatible={report.compatible}"
    )
    return report


@router.post("/align", response_model=ImageAsset, status_code=status.HTTP_201_CREATED)
async def align_rasters(request: AlignmentRequest) -> ImageAsset:
    """
    Reproject and resample a source raster into the coordinate reference system and grid of a reference raster.
    Creates a new derived ImageAsset with explicit provenance tracking.
    """
    try:
        aligned_asset = align_and_reproject_raster(
            source_asset_id=request.source_asset_id,
            reference_asset_id=request.reference_asset_id,
            resampling_name=request.resampling_method
        )
        return aligned_asset
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Alignment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Raster alignment failed: {str(e)}")
