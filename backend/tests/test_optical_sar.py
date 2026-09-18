import os
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from fastapi.testclient import TestClient

from app.main import app
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.domain.modalities import Modality, ImageFormat
from app.schemas.image_asset import ImageAsset
from app.schemas.analysis import AnalysisRequest
from app.db.asset_repository import asset_repository
from app.services.storage import storage_service
from app.services.geospatial.sar_processing import (
    inspect_sar_metadata,
    enhanced_lee_filter,
    preprocess_sar_raster
)
from app.services.geospatial.cross_modal_alignment import (
    calculate_normalized_mutual_information,
    calculate_structural_edge_correlation,
    verify_cross_modal_registration
)
from app.services.models.optical_sar_fusion import (
    OpticalSarFusionRequest,
    optical_sar_fusion_adapter
)
from app.services.orchestration.controller import workflow_controller

client = TestClient(app)


def _create_mock_geotiff(
    filename: str,
    data: np.ndarray,
    crs_epsg: int = 32636,
    bounds: tuple = (300000, 3300000, 301000, 3301000),
    tags: dict = None
) -> str:
    """Helper to write authentic georeferenced GeoTIFF to storage."""
    target_path = storage_service.get_full_path(filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    bands = data.shape[0] if data.ndim == 3 else 1
    h = data.shape[1] if data.ndim == 3 else data.shape[0]
    w = data.shape[2] if data.ndim == 3 else data.shape[1]
    
    transform = from_bounds(*bounds, w, h)
    
    with rasterio.open(
        target_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=str(data.dtype),
        crs=f"EPSG:{crs_epsg}",
        transform=transform
    ) as dst:
        if data.ndim == 3:
            for b in range(bands):
                dst.write(data[b], b + 1)
        else:
            dst.write(data, 1)
        if tags:
            dst.update_tags(**tags)
            
    return target_path


# =========================================================================
# 1. SAR Preprocessing & Physics Tests
# =========================================================================

def test_sar_metadata_inspection_distinguishes_representation():
    """Verify SAR metadata inspection detects calibrated dB, intensity, amplitude and polarization."""
    asset_db = ImageAsset(
        id="sar_db", filename="s.tif", original_filename="s.tif", storage_path="",
        format=ImageFormat.GEOTIFF, modality=Modality.SAR,
        metadata={"SENSOR": "SENTINEL-1", "PRODUCT": "GRD", "SIGMA0_DB": "CALIBRATED", "POLARIZATION": "VV_VH"},
        band_descriptions=["VV", "VH"]
    )
    meta_db = inspect_sar_metadata(asset_db)
    assert meta_db["sensor"] == "SENTINEL-1"
    assert meta_db["product_type"] == "GRD"
    assert "VV" in meta_db["polarization"]
    assert meta_db["input_representation"] == "DB"
    assert meta_db["is_calibrated"] is True

    asset_amp = ImageAsset(
        id="sar_amp", filename="a.tif", original_filename="a.tif", storage_path="",
        format=ImageFormat.GEOTIFF, modality=Modality.SAR,
        metadata={"DN_AMPLITUDE": "TRUE", "MISSION": "RISAT-1"},
        band_descriptions=["Band_1"]
    )
    meta_amp = inspect_sar_metadata(asset_amp)
    assert meta_amp["sensor"] == "RISAT-1"
    assert meta_amp["input_representation"] == "AMPLITUDE"
    assert meta_amp["is_calibrated"] is False


def test_enhanced_lee_filter_variance_reduction():
    """Verify Enhanced Lee filter reduces speckle noise on homogeneous areas while preserving shape."""
    np.random.seed(42)
    # Homogeneous surface with multiplicative speckle noise
    clean = np.ones((64, 64), dtype=np.float32) * 50.0
    speckle = np.random.gamma(shape=1.0, scale=1.0, size=(64, 64)).astype(np.float32)
    noisy = clean * speckle

    # Add a high-contrast point target
    noisy[32, 32] = 500.0

    filtered = enhanced_lee_filter(noisy, window_size=5, damping_factor=1.0)
    assert filtered.shape == noisy.shape
    assert np.all(filtered > 0)

    # Variance in homogeneous patch must be reduced by filtering
    var_noisy = np.var(noisy[:30, :30])
    var_filtered = np.var(filtered[:30, :30])
    assert var_filtered < var_noisy

    # Point target must not be over-smoothed (preserved)
    assert filtered[32, 32] > 200.0


def test_sar_preprocessing_preserves_disk_file():
    """Verify SAR preprocessing converts in-memory to dB but preserves the original file on disk."""
    raw_intensity = np.random.uniform(0.01, 10.0, size=(1, 50, 50)).astype(np.float32)
    filename = "test_raw_sar.tif"
    path = _create_mock_geotiff(filename, raw_intensity)
    
    with open(path, "rb") as f:
        original_bytes = f.read()

    asset = ImageAsset(
        id="sar_raw_001", filename=filename, original_filename=filename,
        storage_path=f"rasters/{filename}", format=ImageFormat.GEOTIFF,
        modality=Modality.SAR, width=50, height=50, bands=1,
        crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000]
    )

    db_out, report = preprocess_sar_raster(raw_intensity, asset, apply_speckle_filter=True)
    assert db_out.shape == (50, 50)
    assert report.speckle_filter_applied == "ENHANCED_LEE_5x5"
    assert report.min_db is not None and report.max_db is not None

    # Disk file remains unchanged
    with open(path, "rb") as f:
        post_bytes = f.read()
    assert original_bytes == post_bytes


# =========================================================================
# 2. Cross-Modal Alignment Tests
# =========================================================================

def test_cross_modal_alignment_high_quality():
    """Verify co-registered optical and SAR imagery achieves HIGH or ACCEPTABLE registration."""
    # Shared edge feature (e.g. water body / river)
    opt = np.ones((60, 60), dtype=np.float32) * 120.0
    sar = np.ones((60, 60), dtype=np.float32) * (-8.0)

    # Water body in lower quadrant: dark in optical, specular low in SAR
    opt[30:, 30:] = 25.0
    sar[30:, 30:] = -22.0

    asset_opt = ImageAsset(
        id="opt_reg_1", filename="opt.tif", original_filename="opt.tif", storage_path="",
        format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=60, height=60, bands=1,
        crs="EPSG:32636", affine_transform=[10, 0, 300000, 0, -10, 3301000],
        geographic_bbox=[31.0, 30.0, 31.01, 30.01]
    )
    asset_sar = ImageAsset(
        id="sar_reg_1", filename="sar.tif", original_filename="sar.tif", storage_path="",
        format=ImageFormat.GEOTIFF, modality=Modality.SAR, width=60, height=60, bands=1,
        crs="EPSG:32636", affine_transform=[10, 0, 300000, 0, -10, 3301000],
        geographic_bbox=[31.0, 30.0, 31.01, 30.01]
    )

    assessment = verify_cross_modal_registration(opt, sar, asset_opt, asset_sar)
    assert assessment.quality_status in ["HIGH", "ACCEPTABLE"]
    assert assessment.suppress_pixel_fusion is False
    assert assessment.normalized_mutual_information > 0.05
    assert assessment.structural_gradient_correlation > 0.10


def test_cross_modal_alignment_poor_quality_suppression():
    """Verify disjoint footprints or mismatched grids trigger POOR status and suppress pixel fusion."""
    opt = np.random.uniform(0, 255, size=(50, 50)).astype(np.float32)
    sar = np.random.uniform(-30, 0, size=(50, 50)).astype(np.float32)

    asset_opt = ImageAsset(
        id="opt_disjoint", filename="opt.tif", original_filename="opt.tif", storage_path="",
        format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=50, height=50, bands=1,
        crs="EPSG:32636", geographic_bbox=[31.0, 30.0, 31.01, 30.01]
    )
    # Disjoint bbox (different region)
    asset_sar = ImageAsset(
        id="sar_disjoint", filename="sar.tif", original_filename="sar.tif", storage_path="",
        format=ImageFormat.GEOTIFF, modality=Modality.SAR, width=50, height=50, bands=1,
        crs="EPSG:32636", geographic_bbox=[35.0, 34.0, 35.01, 34.01]
    )

    assessment = verify_cross_modal_registration(opt, sar, asset_opt, asset_sar)
    assert assessment.quality_status == "POOR"
    assert assessment.suppress_pixel_fusion is True
    assert any("suppressed" in lim.lower() for lim in assessment.limitations)


# =========================================================================
# 3. Optical+SAR Fusion Adapter Tests
# =========================================================================

def test_optical_sar_fusion_agreement_and_null_confidence():
    """Verify optical water + SAR specular reflection produces agreement and confidence=None."""
    opt_data = np.ones((3, 60, 60), dtype=np.uint8) * 140
    # Optical water: low luma (dark) in corner
    opt_data[:, 30:, 30:] = 20

    # SAR data in dB: backscatter ~ -8 dB for land, < -18 dB for calm water
    sar_data = np.ones((1, 60, 60), dtype=np.float32) * (-8.0)
    sar_data[0, 30:, 30:] = -22.0

    opt_file = "opt_water_agree.tif"
    sar_file = "sar_water_agree.tif"
    _create_mock_geotiff(opt_file, opt_data, tags={"COLOR": "RGB"})
    _create_mock_geotiff(sar_file, sar_data, tags={"SENSOR": "SENTINEL-1", "SIGMA0_DB": "CALIBRATED"})

    opt_asset = ImageAsset(
        id="opt_agree_asset", filename=opt_file, original_filename=opt_file,
        storage_path=f"rasters/{opt_file}", format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL, width=60, height=60, bands=3,
        crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
        geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
    )
    sar_asset = ImageAsset(
        id="sar_agree_asset", filename=sar_file, original_filename=sar_file,
        storage_path=f"rasters/{sar_file}", format=ImageFormat.GEOTIFF,
        modality=Modality.SAR, width=60, height=60, bands=1,
        crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
        geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
    )

    asset_repository.save_asset(opt_asset, opt_asset.storage_path)
    asset_repository.save_asset(sar_asset, sar_asset.storage_path)

    req = OpticalSarFusionRequest(
        optical_asset_id=opt_asset.id,
        sar_asset_id=sar_asset.id,
        query="Map water extent from Optical and SAR"
    )
    res = optical_sar_fusion_adapter.predict(req)

    assert res.confidence is None  # Strict zero-fabrication rule

    assert res.uncertainty_state in ["SUPPORTED", "SUPPORTED_WITH_WARNINGS"]
    assert len(res.agreement_regions) > 0
    assert res.agreement_ratio_pct > 10.0
    assert "corroborate" in res.direct_answer.lower() or "confirmed" in res.direct_answer.lower()


def test_optical_sar_fusion_sensor_disagreement():
    """Verify that contradictory signals (e.g. optical dark shadow vs SAR rough terrain) surface divergence."""
    # Optical shows dark surface across region
    opt_data = np.ones((3, 60, 60), dtype=np.uint8) * 140
    opt_data[:, 20:50, 20:50] = 20  # dark patch

    # SAR shows rough/urban double bounce (> -3 dB) across that same patch
    sar_data = np.ones((1, 60, 60), dtype=np.float32) * (-10.0)
    sar_data[0, 20:50, 20:50] = 0.0  # high backscatter (not water)

    opt_file = "opt_disagree.tif"
    sar_file = "sar_disagree.tif"
    _create_mock_geotiff(opt_file, opt_data)
    _create_mock_geotiff(sar_file, sar_data, tags={"SENSOR": "SENTINEL-1", "SIGMA0_DB": "CALIBRATED"})

    opt_asset = ImageAsset(
        id="opt_disagree_asset", filename=opt_file, original_filename=opt_file,
        storage_path=f"rasters/{opt_file}", format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL, width=60, height=60, bands=3,
        crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
        geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
    )
    sar_asset = ImageAsset(
        id="sar_disagree_asset", filename=sar_file, original_filename=sar_file,
        storage_path=f"rasters/{sar_file}", format=ImageFormat.GEOTIFF,
        modality=Modality.SAR, width=60, height=60, bands=1,
        crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
        geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
    )

    asset_repository.save_asset(opt_asset, opt_asset.storage_path)
    asset_repository.save_asset(sar_asset, sar_asset.storage_path)

    req = OpticalSarFusionRequest(
        optical_asset_id=opt_asset.id,
        sar_asset_id=sar_asset.id,
        query="Verify flood extent across sensors"
    )
    res = optical_sar_fusion_adapter.predict(req)

    assert res.confidence is None
    assert len(res.disagreement_regions) > 0
    assert res.disagreement_ratio_pct > 5.0
    assert any("divergence" in f.get("summary", "").lower() for f in res.fused_findings)


# =========================================================================
# 4. Controller End-to-End Orchestration Test
# =========================================================================

def test_controller_optical_sar_end_to_end():
    """Verify workflow controller automatically classifies, plans, executes, and persists Optical+SAR analysis."""
    opt_data = np.ones((3, 50, 50), dtype=np.uint8) * 120
    opt_data[:, 20:, 20:] = 20  # water
    sar_data = np.ones((1, 50, 50), dtype=np.float32) * (-8.0)
    sar_data[0, 20:, 20:] = -20.0  # specular water

    opt_file = "ctrl_opt_fusion.tif"
    sar_file = "ctrl_sar_fusion.tif"
    _create_mock_geotiff(opt_file, opt_data)
    _create_mock_geotiff(sar_file, sar_data, tags={"SENSOR": "SENTINEL-1", "SIGMA0_DB": "CALIBRATED"})

    opt_asset = ImageAsset(
        id="ctrl_opt_fuse_id", filename=opt_file, original_filename=opt_file,
        storage_path=f"rasters/{opt_file}", format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL, width=50, height=50, bands=3,
        crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
        geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
    )
    sar_asset = ImageAsset(
        id="ctrl_sar_fuse_id", filename=sar_file, original_filename=sar_file,
        storage_path=f"rasters/{sar_file}", format=ImageFormat.GEOTIFF,
        modality=Modality.SAR, width=50, height=50, bands=1,
        crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
        geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
    )

    asset_repository.save_asset(opt_asset, opt_asset.storage_path)
    asset_repository.save_asset(sar_asset, sar_asset.storage_path)

    request = AnalysisRequest(
        query="Analyze optical and SAR water extent together",
        image_ids=[opt_asset.id, sar_asset.id]
    )
    result = workflow_controller.execute_analysis(request)

    assert result.status in [AnalysisStatus.COMPLETED, AnalysisStatus.COMPLETED_WITH_WARNINGS]
    assert result.task == TaskFamily.OPTICAL_SAR_ANALYSIS
    assert result.confidence is None
    assert len(result.findings) > 0
    assert result.model_provenance.model_name == "SatQuery-OpticalSAR-FusionAdapter-v1"
    assert len(result.execution_summary) >= 1
    assert result.downloadable_artifacts.report_pdf_url is not None
