import io
import pytest
import numpy as np
import rasterio
from pathlib import Path
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from app.main import app
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.domain.modalities import Modality, ImageFormat
from app.schemas.image_asset import ImageAsset
from app.schemas.analysis import AnalysisRequest
from app.db.asset_repository import asset_repository
from app.registry.model_registry import model_tool_registry
from app.services.geospatial.temporal_validation import validate_temporal_pair, TemporalValidationError
from app.services.geospatial.registration import verify_registration_quality
from app.services.geospatial.area_calculation import compute_pixel_area_m2, calculate_changed_area
from app.services.models.temporal_change import TemporalChangeAdapter
from app.services.models.temporal_vqa import TemporalVqaAdapter
from app.services.models.schemas import TemporalChangeRequest, TemporalVqaRequest
from app.services.orchestration import workflow_controller

client = TestClient(app)

# Repo root is two levels above this file: backend/tests/test_temporal_change.py
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STORAGE_DIR = _REPO_ROOT / "storage"


def create_test_geotiff(
    filename: str,
    width: int = 64,
    height: int = 64,
    bands: int = 3,
    crs: str = "EPSG:32643",
    res: float = 10.0,
    origin: tuple = (500000.0, 2000000.0),
    data_value: float = 100.0
) -> str:
    """Creates a local GeoTIFF file for testing and returns its absolute path."""
    _STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = str(_STORAGE_DIR / filename)
    transform = from_origin(origin[0], origin[1], res, res)
    rng = np.random.RandomState(int(data_value) % 1000 + 42)
    base = rng.uniform(50.0, 180.0, (height, width)).astype(np.float32)
    data = np.stack([base] * bands, axis=0)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=np.float32,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)
    return path



# =========================================================================
# 1. Temporal Input Validation Tests
# =========================================================================

def test_temporal_validation_requires_two_assets():
    """Verify temporal analysis rejects inputs with <2 or >2 images."""
    a1 = ImageAsset(id="a1", filename="a1.tif", original_filename="a1.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)
    with pytest.raises(TemporalValidationError, match="requires exactly 2 images"):
        validate_temporal_pair([a1])

    a2 = ImageAsset(id="a2", filename="a2.tif", original_filename="a2.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)
    a3 = ImageAsset(id="a3", filename="a3.tif", original_filename="a3.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)
    with pytest.raises(TemporalValidationError, match="requires exactly 2 images"):
        validate_temporal_pair([a1, a2, a3])


def test_temporal_validation_modality_compatibility():
    """Verify SAR imagery is rejected in the bi-temporal optical change pipeline."""
    opt = ImageAsset(id="opt", filename="o.tif", original_filename="o.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)
    sar = ImageAsset(id="sar", filename="s.tif", original_filename="s.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.SAR, width=100, height=100, bands=1)
    with pytest.raises(TemporalValidationError, match="Optical-SAR cross-sensor change detection is not supported"):
        validate_temporal_pair([opt, sar])


def test_temporal_ordering_from_metadata_and_fallback():
    """Verify temporal chronology resolution from ISO timestamps or explicit user ordering."""
    a1 = ImageAsset(
        id="asset_early",
        filename="e.tif",
        original_filename="e.tif",
        storage_path="",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=100,
        height=100,
        bands=3,
        acquisition_time="2021-05-01T10:00:00Z",
        crs="EPSG:4326",
        geographic_bbox=[77.0, 12.0, 77.1, 12.1]
    )
    a2 = ImageAsset(
        id="asset_late",
        filename="l.tif",
        original_filename="l.tif",
        storage_path="",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=100,
        height=100,
        bands=3,
        acquisition_time="2023-08-15T10:00:00Z",
        crs="EPSG:4326",
        geographic_bbox=[77.0, 12.0, 77.1, 12.1]
    )

    # Automatically orders by timestamp
    t1, t2, meta = validate_temporal_pair([a2, a1])
    assert t1.id == "asset_early"
    assert t2.id == "asset_late"
    assert meta["order_provenance"] == "metadata"

    # Respects explicit user-specified override
    t1_u, t2_u, meta_u = validate_temporal_pair([a1, a2], user_order=["asset_late", "asset_early"])
    assert t1_u.id == "asset_late"
    assert t2_u.id == "asset_early"
    assert meta_u["order_provenance"] == "user_specified"


def test_temporal_non_overlapping_rejection():
    """Verify disjoint spatial footprints are rejected."""
    a1 = ImageAsset(id="d1", filename="d1.tif", original_filename="d1.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3, crs="EPSG:4326", geographic_bbox=[10.0, 10.0, 11.0, 11.0])
    a2 = ImageAsset(id="d2", filename="d2.tif", original_filename="d2.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3, crs="EPSG:4326", geographic_bbox=[20.0, 20.0, 21.0, 21.0])
    with pytest.raises(TemporalValidationError, match="no geographic spatial intersection"):
        validate_temporal_pair([a1, a2])


# =========================================================================
# 2. Registration Verification Tests
# =========================================================================

def test_registration_verification_high_quality():
    """Verify identical or subpixel-shifted images receive HIGH registration status."""
    img1 = np.random.RandomState(42).uniform(0, 255, (64, 64)).astype(np.float32)
    img2 = img1.copy()
    
    assessment = verify_registration_quality(img1, img2)
    assert assessment.quality_status == "HIGH"
    assert assessment.shift_magnitude_pixels <= 0.5
    assert assessment.correlation_coefficient >= 0.95
    assert assessment.is_sufficient_for_pixel_localization is True


def test_registration_verification_poor_quality_warning():
    """Verify uncorrelated images receive POOR registration quality and warn against fine localization."""
    img1 = np.random.RandomState(10).uniform(0, 255, (64, 64)).astype(np.float32)
    img2 = np.random.RandomState(99).uniform(0, 255, (64, 64)).astype(np.float32)
    
    assessment = verify_registration_quality(img1, img2)
    assert assessment.quality_status == "POOR"
    assert assessment.correlation_coefficient < 0.20
    assert assessment.is_sufficient_for_pixel_localization is False
    assert any("co-registration residual" in w for w in assessment.warnings)


# =========================================================================
# 3. Area Calculation & Coordinate Projection Tests
# =========================================================================

def test_area_calculation_projected_utm():
    """Verify projected metric CRS calculates dx * dy pixel area."""
    # 10m resolution in UTM -> 100 m² per pixel
    area_m2 = compute_pixel_area_m2(crs_str="EPSG:32643", resolution=10.0)
    assert area_m2 == 100.0

    stats = calculate_changed_area(num_changed_pixels=100, crs_str="EPSG:32643", resolution=10.0)
    assert stats["sq_meters"] == 10000.0
    assert stats["hectares"] == 1.0
    assert stats["sq_km"] == 0.01


def test_area_calculation_geographic_ellipsoidal():
    """Verify geographic CRS (EPSG:4326) computes true geodesic metric area, not degrees squared."""
    # At latitude 28°N (Delhi), 0.0001° is ~11.1m lat and ~9.8m lon -> ~109 m²
    affine = [0.0001, 0.0, 77.0, 0.0, -0.0001, 28.0]
    area_m2 = compute_pixel_area_m2(crs_str="EPSG:4326", center_lat=28.0, affine_transform=affine)
    
    # Must be metric ground area (~109 m²), NEVER 0.00000001 degrees²!
    assert 90.0 < area_m2 < 130.0

    stats = calculate_changed_area(num_changed_pixels=1000, crs_str="EPSG:4326", center_lat=28.0, affine_transform=affine)
    assert stats["sq_meters"] > 90000.0
    assert stats["hectares"] > 9.0


# =========================================================================
# 4. Specialist Temporal Change Model Tests
# =========================================================================

def test_temporal_change_identical_baseline_no_change():
    """Verify identical image pair returns NO_CHANGE_DETECTED with 0 changed area."""
    t1_path = create_test_geotiff("test_nochange_t1.tif", width=64, height=64, data_value=120.0)
    asset_t1 = ImageAsset(
        id="t1_identical",
        filename="test_nochange_t1.tif",
        original_filename="t1.tif",
        storage_path="test_nochange_t1.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=64,
        height=64,
        bands=3,
        crs="EPSG:32643",
        resolution=10.0,
        geographic_bbox=[77.0, 12.0, 77.01, 12.01],
        validation_status="valid"
    )
    asset_t2 = ImageAsset(
        id="t2_identical",
        filename="test_nochange_t1.tif",  # same path
        original_filename="t2.tif",
        storage_path="test_nochange_t1.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=64,
        height=64,
        bands=3,
        crs="EPSG:32643",
        resolution=10.0,
        geographic_bbox=[77.0, 12.0, 77.01, 12.01],
        validation_status="valid"
    )
    asset_repository.save_asset(asset_t1, t1_path)
    asset_repository.save_asset(asset_t2, t1_path)

    adapter = TemporalChangeAdapter(device="cpu")
    req = TemporalChangeRequest(t1_asset_id=asset_t1.id, t2_asset_id=asset_t2.id)
    res = adapter.predict(req)

    assert res.change_verdict == "NO_CHANGE_DETECTED"
    assert res.total_changed_pixels == 0
    assert res.changed_area_ha == 0.0
    assert res.confidence is None
    assert len(res.change_clusters) == 0


def test_temporal_change_cva_synthetic_fixture():
    """Verify Change Vector Analysis detects distinct synthetic change blocks."""
    # Create T1 with background 50
    t1_path = create_test_geotiff("test_cva_t1.tif", width=64, height=64, data_value=50.0)
    # Create T2 with background 50 and a 20x20 block of 220 in the center
    t2_path = str(_STORAGE_DIR / "test_cva_t2.tif")
    with rasterio.open(t1_path) as src:
        profile = src.profile.copy()
        data2 = src.read().astype(np.float32)
        data2[:, 20:40, 20:40] = 220.0  # Synthetic change cluster
        with rasterio.open(t2_path, "w", **profile) as dst:
            dst.write(data2)

    a1 = ImageAsset(id="a_cva_1", filename="test_cva_t1.tif", original_filename="t1.tif", storage_path="test_cva_t1.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=64, height=64, bands=3, crs="EPSG:32643", resolution=10.0, geographic_bbox=[77.0, 12.0, 77.01, 12.01], validation_status="valid")
    a2 = ImageAsset(id="a_cva_2", filename="test_cva_t2.tif", original_filename="t2.tif", storage_path="test_cva_t2.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=64, height=64, bands=3, crs="EPSG:32643", resolution=10.0, geographic_bbox=[77.0, 12.0, 77.01, 12.01], validation_status="valid")
    asset_repository.save_asset(a1, t1_path)
    asset_repository.save_asset(a2, t2_path)

    adapter = TemporalChangeAdapter(device="cpu")
    req = TemporalChangeRequest(t1_asset_id=a1.id, t2_asset_id=a2.id, method="analytical_cva")
    res = adapter.predict(req)

    assert res.change_verdict in ["OBSERVED_CHANGE", "POSSIBLE_CHANGE"]
    assert res.total_changed_pixels > 0
    assert len(res.change_clusters) >= 1
    # Check top cluster location corresponds to center block [20, 20, 40, 40]
    top_cluster = res.change_clusters[0]
    assert 15 <= top_cluster.pixel_box[0] <= 25
    assert 35 <= top_cluster.pixel_box[2] <= 45


# =========================================================================
# 5. Temporal Change VQA & Semantic Boundary Tests
# =========================================================================

def test_temporal_vqa_semantic_boundary_safeguard():
    """Verify Change VQA explicitly states semantic limitations when asked about built-up or water classes."""
    t1_path = create_test_geotiff("test_vqa_sem_t1.tif", width=64, height=64, data_value=50.0)
    t2_path = str(_STORAGE_DIR / "test_vqa_sem_t2.tif")
    with rasterio.open(t1_path) as src:
        data = src.read().astype(np.float32)
        data[:, 10:30, 10:30] = 200.0
        with rasterio.open(t2_path, "w", **src.profile) as dst:
            dst.write(data)

    a1 = ImageAsset(id="sem_a1", filename="test_vqa_sem_t1.tif", original_filename="t1.tif", storage_path="test_vqa_sem_t1.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=64, height=64, bands=3, crs="EPSG:32643", resolution=10.0, geographic_bbox=[77.0, 12.0, 77.01, 12.01], validation_status="valid")
    a2 = ImageAsset(id="sem_a2", filename="test_vqa_sem_t2.tif", original_filename="t2.tif", storage_path="test_vqa_sem_t2.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=64, height=64, bands=3, crs="EPSG:32643", resolution=10.0, geographic_bbox=[77.0, 12.0, 77.01, 12.01], validation_status="valid")
    asset_repository.save_asset(a1, t1_path)
    asset_repository.save_asset(a2, t2_path)

    vqa_adapter = TemporalVqaAdapter()
    req = TemporalVqaRequest(
        t1_asset_id=a1.id,
        t2_asset_id=a2.id,
        question="Has the built-up area increased?",
        parameters={"method": "analytical_cva"}
    )
    res = vqa_adapter.predict(req)

    # SIH non-negotiable requirement: Must NOT claim 'built-up increased' without a semantic segmenter
    assert "semantic" in res.answer.lower()
    assert "cannot be conclusively confirmed" in res.answer.lower()
    assert any("Semantic boundary limitation" in w for w in res.warnings)
    assert res.confidence is None


def test_temporal_vqa_location_query():
    """Verify Change VQA answers where change occurred by pinpointing spatial quadrants."""
    a1 = asset_repository.get_asset("sem_a1")
    a2 = asset_repository.get_asset("sem_a2")

    vqa_adapter = TemporalVqaAdapter()
    req = TemporalVqaRequest(
        t1_asset_id=a1.id,
        t2_asset_id=a2.id,
        question="Where did the change occur?",
        parameters={"method": "analytical_cva"}
    )
    res = vqa_adapter.predict(req)
    assert "cluster" in res.answer.lower()
    assert len(res.change_clusters) >= 1


# =========================================================================
# 6. Controller & Tool Registry Integration Tests
# =========================================================================

def test_registry_has_activated_temporal_tools():
    """Verify adapter_temporal_change and adapter_temporal_change_vqa are registered as AVAILABLE."""
    tool_change = model_tool_registry.get_tool("adapter_temporal_change")
    assert tool_change is not None
    assert tool_change.enabled is True
    assert tool_change.availability == "AVAILABLE"
    assert tool_change.model_name in ["satquery/attentionchangenet-v1", "HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff"]

    tool_vqa = model_tool_registry.get_tool("adapter_temporal_change_vqa")
    assert tool_vqa is not None
    assert tool_vqa.enabled is True
    assert tool_vqa.availability == "AVAILABLE"


def test_controller_temporal_end_to_end():
    """Verify workflow_controller automatically plans and executes bi-temporal change analysis."""
    a1 = asset_repository.get_asset("sem_a1")
    a2 = asset_repository.get_asset("sem_a2")

    req = AnalysisRequest(
        query="What changed between these two dates?",
        image_ids=[a1.id, a2.id],
        parameters={"method": "analytical_cva"}
    )
    result = workflow_controller.execute_analysis(req)

    assert result.status == AnalysisStatus.COMPLETED
    assert result.task == TaskFamily.TEMPORAL_CHANGE_VQA
    assert "surface change" in result.answer.lower() or "change" in result.answer.lower()
    assert result.confidence is None
    assert len(result.execution_summary) >= 1
    assert result.execution_summary[0].tool_name in ["satquery/temporal-change-vqa-engine", "adapter_temporal_change_vqa"]
    assert result.execution_summary[0].status == "completed"
    assert result.evidence_assessment is not None
    assert result.evidence_assessment.verification_method == "bitemporal_siamese_and_phase_correlation_audit"
