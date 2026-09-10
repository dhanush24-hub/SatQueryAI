import io
from fastapi.testclient import TestClient
from app.main import app
from app.domain.tasks import AnalysisStatus
from app.domain.modalities import ImageFormat, Modality

client = TestClient(app)


from PIL import Image
from tests.test_geospatial import create_in_memory_geotiff


def test_upload_images():
    # Ingest a valid GeoTIFF and benchmark PNG
    geotiff_bytes = create_in_memory_geotiff(width=64, height=64, bands=1, tags={"SENSOR": "RISAT-1", "POLARIZATION": "VV"})
    
    png_buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 150, 200)).save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    files = [
        ("files", ("scene1.tif", io.BytesIO(geotiff_bytes), "image/tiff")),
        ("files", ("benchmark_scene.png", io.BytesIO(png_bytes), "image/png"))
    ]
    response = client.post("/api/uploads", files=files, data={"benchmark_mode": "true"})
    assert response.status_code == 201
    data = response.json()
    assert len(data) == 2
    assert data[0]["format"] == ImageFormat.GEOTIFF
    assert data[0]["validation_status"] == "valid"
    assert data[1]["format"] == ImageFormat.PNG
    assert data[1]["validation_status"] == "valid"


def test_upload_unsupported_format():
    files = [
        ("files", ("malicious.exe", io.BytesIO(b"MZ\x90\x00_NOT_A_VALID_RASTER"), "application/x-msdownload"))
    ]
    response = client.post("/api/uploads", files=files)
    assert response.status_code in (400, 415)


def test_analysis_endpoint_not_implemented():
    # Analysis with invalid asset ID returns 404
    payload = {
        "query": "Is there flooding along the river perimeter?",
        "image_ids": ["img_nonexistent_001"]
    }
    response = client.post("/api/analysis", json=payload)
    assert response.status_code == 404

    # Analysis with SAR asset on optical models returns explicit MODEL_UNAVAILABLE status
    from app.schemas.image_asset import ImageAsset
    from app.db.asset_repository import asset_repository
    sar_asset = ImageAsset(
        id="sar_asset_endpoint_test",
        filename="sar_test.tif",
        original_filename="sar.tif",
        storage_path="mock/path.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.SAR,
        width=100,
        height=100,
        bands=1
    )
    asset_repository.save_asset(sar_asset, "mock/path.tif")

    res2 = client.post("/api/analysis", json={"query": "Check flooding", "image_ids": [sar_asset.id]})
    assert res2.status_code == 200
    data = res2.json()
    assert data["status"] == AnalysisStatus.MODEL_UNAVAILABLE
    assert "sar" in data["answer"].lower()
    assert len(data["evidence"]) == 0
    assert len(data["warnings"]) >= 1


def test_jobs_endpoint_not_found():
    response = client.get("/api/jobs/job_nonexistent")
    assert response.status_code == 404
