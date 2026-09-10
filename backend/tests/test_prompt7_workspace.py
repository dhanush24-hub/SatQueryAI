import os
import io
import json
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.domain.modalities import Modality, ImageFormat
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.schemas.image_asset import ImageAsset
from app.schemas.analysis import AnalysisRequest, AnalysisResult, FindingItem, BoundingBox
from app.db.asset_repository import asset_repository
from app.db.analysis_repository import analysis_repository
from app.services.storage.local import storage_service
from app.services.reporting.pdf_generator import pdf_generator
from app.services.orchestration.controller import workflow_controller


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def synthetic_referenced_assets():
    """Create a pair of synthetically referenced GeoTIFF assets with authentic UTM CRS."""
    t1_id = "test_ref_t1"
    t2_id = "test_ref_t2"

    np.random.seed(42)
    base = (np.random.rand(100, 100) * 120 + 50).astype(np.uint8)
    arr1 = np.stack([base, base, base], axis=0)
    arr2 = arr1.copy()
    arr2[:, 20:50, 20:50] = 230  # distinct square change

    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(500000, 2000000, 10, 10)
    crs = "EPSG:32643"

    for aid, arr in [(t1_id, arr1), (t2_id, arr2)]:
        path = storage_service.get_full_path(f"{aid}.tif")
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=100, width=100,
            count=3, dtype=arr.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(arr)

        asset = ImageAsset(
            id=aid,
            filename=f"{aid}.tif",
            storage_path=path,
            original_filename=f"{aid}.tif",
            format=ImageFormat.TIFF,
            modality=Modality.OPTICAL,
            width=100,
            height=100,
            bands=3,
            crs=crs,
            resolution=10.0,
            geographic_bbox=[75.0, 15.0, 75.01, 15.01],
            sensor="Synthetic-UTM",
            validation_status="benchmark",
            metadata={"benchmark_mode": True}
        )
        asset_repository.save_asset(asset, internal_storage_path=path)

    return t1_id, t2_id


@pytest.fixture
def synthetic_unreferenced_assets():
    """Create a pair of unreferenced PNG benchmark assets without CRS (identity transform)."""
    t1_id = "test_unref_t1"
    t2_id = "test_unref_t2"

    np.random.seed(42)
    base = (np.random.rand(100, 100) * 120 + 50).astype(np.uint8)
    img1_arr = np.stack([base, base, base], axis=-1)
    img2_arr = img1_arr.copy()
    img2_arr[30:60, 30:60, :] = 230

    img1 = Image.fromarray(img1_arr)
    img2 = Image.fromarray(img2_arr)

    for aid, img in [(t1_id, img1), (t2_id, img2)]:
        path = storage_service.get_full_path(f"{aid}.png")
        img.save(path, format="PNG")

        asset = ImageAsset(
            id=aid,
            filename=f"{aid}.png",
            storage_path=path,
            original_filename=f"{aid}.png",
            format=ImageFormat.PNG,
            modality=Modality.OPTICAL,
            width=100,
            height=100,
            bands=3,
            crs=None,  # strictly unreferenced
            resolution=None,
            geographic_bbox=None,
            sensor="Optical-RGB",
            validation_status="benchmark",
            metadata={"benchmark_mode": True}
        )
        asset_repository.save_asset(asset, internal_storage_path=path)

    return t1_id, t2_id


def test_sqlite_analysis_persistence_and_history(synthetic_unreferenced_assets):
    """Verify AnalysisRepository saves and reloads analyses without re-running models."""
    t1_id, t2_id = synthetic_unreferenced_assets
    req = AnalysisRequest(
        query="What changed between these two dates?",
        image_ids=[t1_id, t2_id]
    )
    result = workflow_controller.execute_analysis(req)
    analysis_repository.save_analysis(result, req)

    # Reload from SQLite
    loaded = analysis_repository.get_analysis(result.id)
    assert loaded is not None
    assert loaded.id == result.id
    assert loaded.answer == result.answer
    assert loaded.evidence_state == result.evidence_state
    assert len(loaded.findings) == len(result.findings)

    # Check history summary listing
    history = analysis_repository.list_analyses(limit=10)
    assert any(h.id == result.id for h in history)
    matching = next(h for h in history if h.id == result.id)
    assert matching.query == req.query


def test_area_display_rules_unreferenced_vs_referenced(synthetic_referenced_assets, synthetic_unreferenced_assets):
    """Zero-Fabrication Policy: Area and GeoJSON polygon must be None/omitted for unreferenced imagery."""
    # 1. Unreferenced Pair
    unref_t1, unref_t2 = synthetic_unreferenced_assets
    req_unref = AnalysisRequest(
        query="What changed between these two dates?",
        image_ids=[unref_t1, unref_t2],
        parameters={"method": "analytical_cva"}
    )
    res_unref = workflow_controller.execute_analysis(req_unref)
    assert res_unref.overlays is not None
    assert res_unref.overlays.georeferenced is False
    assert any("Area calculation omitted" in w for w in res_unref.warnings)
    # Finding areas must be None
    for f in res_unref.findings:
        assert f.area_m2 is None
        assert f.geometry is None

    # 2. Referenced Pair
    ref_t1, ref_t2 = synthetic_referenced_assets
    req_ref = AnalysisRequest(
        query="What changed between these two dates?",
        image_ids=[ref_t1, ref_t2],
        parameters={"method": "analytical_cva"}
    )
    res_ref = workflow_controller.execute_analysis(req_ref)
    assert res_ref.overlays is not None
    assert res_ref.overlays.georeferenced is True
    assert res_ref.overlays.total_changed_pixels is not None
    assert len(res_ref.findings) > 0
    first_finding = res_ref.findings[0]
    assert first_finding.area_m2 is not None
    assert first_finding.area_m2 > 0.0
    assert first_finding.geometry is not None
    assert first_finding.geometry["type"] == "Polygon"


def test_reportlab_pdf_report_generation(synthetic_referenced_assets):
    """Verify real PDF report is generated with valid %PDF header and non-zero size."""
    ref_t1, ref_t2 = synthetic_referenced_assets
    assets = [asset_repository.get_asset(ref_t1), asset_repository.get_asset(ref_t2)]
    req = AnalysisRequest(query="What changed between these two dates?", image_ids=[ref_t1, ref_t2], parameters={"method": "analytical_cva"})
    result = workflow_controller.execute_analysis(req)

    filename = f"test_report_{result.id}.pdf"
    gen_file = pdf_generator.generate_report(result, assets, output_filename=filename)
    full_path = storage_service.get_full_path(gen_file)

    assert os.path.exists(full_path)
    file_size = os.path.getsize(full_path)
    assert file_size > 1000  # More than 1 KB

    with open(full_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"


def test_export_geojson_restrictions(client, synthetic_referenced_assets, synthetic_unreferenced_assets):
    """GeoJSON endpoint must return HTTP 400 for unreferenced and HTTP 200 with FeatureCollection for referenced."""
    # 1. Unreferenced analysis
    unref_t1, unref_t2 = synthetic_unreferenced_assets
    res1 = client.post("/api/analysis", json={"query": "What changed?", "image_ids": [unref_t1, unref_t2], "parameters": {"method": "analytical_cva"}})
    assert res1.status_code == 200
    anl_unref_id = res1.json()["id"]

    # Export geojson on unreferenced must fail with 400
    geo_res1 = client.get(f"/api/analysis/{anl_unref_id}/export.geojson")
    assert geo_res1.status_code == 400
    assert "authentic georeferencing" in geo_res1.json()["detail"]

    # 2. Referenced analysis
    ref_t1, ref_t2 = synthetic_referenced_assets
    res2 = client.post("/api/analysis", json={"query": "What changed?", "image_ids": [ref_t1, ref_t2], "parameters": {"method": "analytical_cva"}})
    assert res2.status_code == 200
    anl_ref_id = res2.json()["id"]

    # Export geojson on referenced must succeed with FeatureCollection
    geo_res2 = client.get(f"/api/analysis/{anl_ref_id}/export.geojson")
    assert geo_res2.status_code == 200
    doc = geo_res2.json()
    assert doc["type"] == "FeatureCollection"
    assert len(doc["features"]) > 0
    assert doc["features"][0]["geometry"]["type"] == "Polygon"


def test_export_json_and_pdf_endpoints(client, synthetic_referenced_assets):
    """Verify GET /api/analysis/{id}/export.json and GET /api/analysis/{id}/report.pdf endpoints."""
    ref_t1, ref_t2 = synthetic_referenced_assets
    res = client.post("/api/analysis", json={"query": "What changed?", "image_ids": [ref_t1, ref_t2]})
    assert res.status_code == 200
    anl_id = res.json()["id"]

    # JSON export
    json_res = client.get(f"/api/analysis/{anl_id}/export.json")
    assert json_res.status_code == 200
    assert json_res.headers["content-type"] == "application/json"
    parsed = json.loads(json_res.content)
    assert parsed["id"] == anl_id

    # PDF export
    pdf_res = client.get(f"/api/analysis/{anl_id}/report.pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF-")


def test_unsupported_semantic_transition_safeguard(synthetic_unreferenced_assets):
    """Unsupported transition queries ('Was forest converted to buildings?') must produce explicit refusal message."""
    t1_id, t2_id = synthetic_unreferenced_assets
    req = AnalysisRequest(
        query="Was forest converted to buildings in this area?",
        image_ids=[t1_id, t2_id]
    )
    result = workflow_controller.execute_analysis(req)

    # Must contain the mandated safeguard sentence
    assert "Change was detected in these regions, but the active model cannot reliably determine the land-cover transition type." in result.answer
    assert result.evidence_state == "SUPPORTED_WITH_WARNINGS"
    assert any("Semantic transition" in w for w in result.warnings)


def test_execution_trace_structure_and_timing(synthetic_referenced_assets):
    """Verify observable execution trace contains tool_name, status, summary, and duration."""
    ref_t1, ref_t2 = synthetic_referenced_assets
    req = AnalysisRequest(query="What changed between these two dates?", image_ids=[ref_t1, ref_t2])
    result = workflow_controller.execute_analysis(req)

    assert len(result.execution_summary) > 0
    for step in result.execution_summary:
        assert step.tool_name is not None
        assert step.status in ["completed", "failed", "running"]
        assert len(step.summary) > 0
        assert step.duration_ms is not None
        assert step.duration_ms >= 0.0


def test_single_image_vqa_and_grounding_evidence_structure():
    """Verify single-image VQA and Grounding preserve the same structured evidence format."""
    img = Image.new("RGB", (100, 100), (80, 120, 80))
    path = storage_service.get_full_path("test_single_img.png")
    img.save(path, format="PNG")

    asset = ImageAsset(
        id="test_single_asset",
        filename="test_single_img.png",
        storage_path=path,
        original_filename="test_single_img.png",
        format=ImageFormat.PNG,
        modality=Modality.OPTICAL,
        width=100,
        height=100,
        bands=3,
        sensor="Optical-RGB"
    )
    asset_repository.save_asset(asset, internal_storage_path=path)

    # 1. Single VQA
    vqa_req = AnalysisRequest(query="What is the primary color in this image?", image_ids=["test_single_asset"])
    vqa_res = workflow_controller.execute_analysis(vqa_req)
    assert vqa_res.task == TaskFamily.SINGLE_VQA
    assert len(vqa_res.findings) >= 1
    assert vqa_res.findings[0].type == "vqa_observation"
    assert vqa_res.evidence_state in ["SUPPORTED", "SUPPORTED_WITH_WARNINGS"]

    # 2. Single Grounding
    ground_req = AnalysisRequest(query="Locate the green patch", image_ids=["test_single_asset"])
    ground_res = workflow_controller.execute_analysis(ground_req)
    assert ground_res.task == TaskFamily.SINGLE_GROUNDING
    assert ground_res.evidence_state in ["SUPPORTED", "SUPPORTED_WITH_WARNINGS", "WEAK", "INSUFFICIENT_EVIDENCE"]
