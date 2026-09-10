import io
import os
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient
from app.main import app
from app.domain.modalities import ImageFormat, Modality
from app.schemas.compatibility import PairType
from app.db.asset_repository import AssetRepository
from app.db.session import init_db

client = TestClient(app)


def create_in_memory_geotiff(
    width: int = 100,
    height: int = 100,
    bands: int = 3,
    crs: str = "EPSG:32644",
    west: float = 300000.0,
    north: float = 1400000.0,
    resolution: float = 10.0,
    dtype: str = "uint8",
    nodata: float = 0.0,
    tags: dict = None
) -> bytes:
    """Helper to generate a real, valid GeoTIFF in memory with Rasterio."""
    transform = from_origin(west, north, resolution, resolution)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }

    memfile = io.BytesIO()
    with rasterio.open(memfile, "w", **profile) as dst:
        for b in range(1, bands + 1):
            data = (np.ones((height, width), dtype=dtype) * (b * 50)).astype(dtype)
            dst.write(data, b)
        if tags:
            dst.update_tags(**tags)

    return memfile.getvalue()


def test_geotiff_upload_and_metadata_extraction():
    # 1. Create a 3-band GeoTIFF with UTM CRS
    tiff_bytes = create_in_memory_geotiff(
        width=200,
        height=150,
        bands=3,
        crs="EPSG:32644",
        resolution=10.0,
        tags={"SATELLITE": "Cartosat-2S", "TIFFTAG_DATETIME": "2026-04-15 08:30:00"}
    )

    files = [("files", ("hyderabad_cartosat.tif", io.BytesIO(tiff_bytes), "image/tiff"))]
    response = client.post("/api/uploads", files=files)
    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1

    asset = data[0]
    assert asset["format"] == ImageFormat.GEOTIFF
    assert asset["width"] == 200
    assert asset["height"] == 150
    assert asset["bands"] == 3
    assert asset["crs"] == "EPSG:32644"
    assert asset["resolution"] == 10.0
    assert asset["sensor"] == "Cartosat-2S"
    assert asset["acquisition_time"] == "2026-04-15 08:30:00"
    assert asset["has_preview"] is True
    assert asset["preview_url"] == f"/api/uploads/{asset['id']}/preview"

    # CRITICAL: Verify absolute server storage path is NOT leaked
    assert not asset["storage_path"].startswith("/")
    assert asset["storage_path"].startswith("storage/")


def test_corrupted_tiff_rejection():
    # Fake TIFF header followed by random garbage
    corrupt_bytes = b"\x49\x49\x2a\x00\xff\xff\xff\xff\x00\x00corrupted_payload_bytes"
    files = [("files", ("corrupted.tif", io.BytesIO(corrupt_bytes), "image/tiff"))]
    response = client.post("/api/uploads", files=files)
    assert response.status_code == 400
    assert "corrupt or not a readable geospatial raster" in response.json()["detail"].lower()


def test_unsupported_file_signature():
    # Plain text file
    txt_bytes = b"Hello world this is not a raster"
    files = [("files", ("notes.txt", io.BytesIO(txt_bytes), "text/plain"))]
    response = client.post("/api/uploads", files=files)
    assert response.status_code == 400
    assert "unsupported or invalid raster file signature" in response.json()["detail"].lower()


def test_png_benchmark_mode_gating():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(buf, format="PNG")
    png_bytes = buf.getvalue()
    
    # Attempt 1: Upload PNG without benchmark_mode -> Must be rejected per SIH GeoTIFF-first rule
    files = [("files", ("benchmark_sample.png", io.BytesIO(png_bytes), "image/png"))]
    response = client.post("/api/uploads", files=files)
    assert response.status_code == 400
    assert "benchmark evaluation mode" in response.json()["detail"].lower()

    # Attempt 2: Upload PNG with benchmark_mode=True -> Accepted
    files = [("files", ("benchmark_sample.png", io.BytesIO(png_bytes), "image/png"))]
    data_form = {"benchmark_mode": "true"}
    response = client.post("/api/uploads", files=files, data=data_form)
    assert response.status_code == 201
    asset = response.json()[0]
    assert asset["format"] == ImageFormat.PNG


def test_preview_generation_endpoint():
    # Upload single band raster
    tiff_bytes = create_in_memory_geotiff(width=64, height=64, bands=1, crs="EPSG:4326")
    files = [("files", ("sar_scene.tif", io.BytesIO(tiff_bytes), "image/tiff"))]
    res = client.post("/api/uploads", files=files)
    assert res.status_code == 201
    asset_id = res.json()[0]["id"]

    # Fetch preview
    preview_res = client.get(f"/api/uploads/{asset_id}/preview")
    assert preview_res.status_code == 200
    assert preview_res.headers["content-type"] == "image/png"
    assert preview_res.headers.get("cache-control") == "public, max-age=86400"
    assert len(preview_res.content) > 50  # Valid PNG image


def test_sqlite_persistence_across_reloads():
    # Test persistent storage in SQLite
    temp_db = tempfile.mktemp(suffix=".db")
    init_db(temp_db)
    repo = AssetRepository(db_path=temp_db)

    tiff_bytes = create_in_memory_geotiff(width=50, height=50, bands=1)
    files = [("files", ("persist_test.tif", io.BytesIO(tiff_bytes), "image/tiff"))]
    res = client.post("/api/uploads", files=files)
    asset_id = res.json()[0]["id"]

    # Fetch via API
    get_res = client.get(f"/api/uploads/{asset_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == asset_id
    assert get_res.json()["width"] == 50


def test_pair_compatibility_overlapping():
    # Scene A: UTM 32644 (Hyderabad)
    tiff_a = create_in_memory_geotiff(
        width=100, height=100, crs="EPSG:32644",
        west=300000.0, north=1400000.0, resolution=10.0,
        tags={"TIFFTAG_DATETIME": "2026-01-01"}
    )
    # Scene B: Exactly overlapping UTM 32644
    tiff_b = create_in_memory_geotiff(
        width=100, height=100, crs="EPSG:32644",
        west=300000.0, north=1400000.0, resolution=10.0,
        tags={"TIFFTAG_DATETIME": "2026-02-01"}
    )

    res_a = client.post("/api/uploads", files=[("files", ("a.tif", io.BytesIO(tiff_a), "image/tiff"))]).json()[0]
    res_b = client.post("/api/uploads", files=[("files", ("b.tif", io.BytesIO(tiff_b), "image/tiff"))]).json()[0]

    # Check compatibility for TEMPORAL
    compat_payload = {
        "asset_id_a": res_a["id"],
        "asset_id_b": res_b["id"],
        "pair_type": "TEMPORAL"
    }
    compat_res = client.post("/api/compatibility/check", json=compat_payload)
    assert compat_res.status_code == 200
    report = compat_res.json()

    assert report["compatible"] is True
    assert report["crs_match"] is True
    assert report["has_spatial_overlap"] is True
    assert report["spatial_overlap_percentage"] > 99.0
    assert report["requires_reprojection"] is False
    assert report["is_coregistered"] is True


def test_pair_compatibility_disjoint():
    # Scene A: Bangalore
    tiff_a = create_in_memory_geotiff(
        width=50, height=50, crs="EPSG:4326",
        west=77.5, north=13.0, resolution=0.001
    )
    # Scene B: Delhi (completely disjoint)
    tiff_b = create_in_memory_geotiff(
        width=50, height=50, crs="EPSG:4326",
        west=77.2, north=28.6, resolution=0.001
    )

    res_a = client.post("/api/uploads", files=[("files", ("bangalore.tif", io.BytesIO(tiff_a), "image/tiff"))]).json()[0]
    res_b = client.post("/api/uploads", files=[("files", ("delhi.tif", io.BytesIO(tiff_b), "image/tiff"))]).json()[0]

    compat_payload = {
        "asset_id_a": res_a["id"],
        "asset_id_b": res_b["id"],
        "pair_type": "TEMPORAL"
    }
    compat_res = client.post("/api/compatibility/check", json=compat_payload)
    assert compat_res.status_code == 200
    report = compat_res.json()

    assert report["compatible"] is False
    assert report["has_spatial_overlap"] is False
    assert report["spatial_overlap_percentage"] == 0.0
    assert any("disjoint" in w.lower() for w in report["warnings"])


def test_raster_alignment_and_reprojection():
    # Scene A in EPSG:32644 (100x100)
    tiff_a = create_in_memory_geotiff(
        width=100, height=100, crs="EPSG:32644",
        west=300000.0, north=1400000.0, resolution=10.0
    )
    # Scene B overlapping in EPSG:4326
    # 300000, 1400000 in UTM 44N corresponds approximately to ~73.1°E, 12.65°N
    tiff_b = create_in_memory_geotiff(
        width=120, height=120, crs="EPSG:32644",
        west=300100.0, north=1400100.0, resolution=10.0
    )

    res_a = client.post("/api/uploads", files=[("files", ("target_grid.tif", io.BytesIO(tiff_a), "image/tiff"))]).json()[0]
    res_b = client.post("/api/uploads", files=[("files", ("to_align.tif", io.BytesIO(tiff_b), "image/tiff"))]).json()[0]

    align_payload = {
        "source_asset_id": res_b["id"],
        "reference_asset_id": res_a["id"],
        "resampling_method": "bilinear"
    }
    align_res = client.post("/api/compatibility/align", json=align_payload)
    assert align_res.status_code == 201
    aligned_asset = align_res.json()

    # The aligned raster should now have the exact dimensions and CRS of the reference grid
    assert aligned_asset["width"] == res_a["width"]
    assert aligned_asset["height"] == res_a["height"]
    assert aligned_asset["crs"] == res_a["crs"]
    assert "provenance" in aligned_asset["metadata"]
    assert aligned_asset["metadata"]["provenance"]["source_asset_id"] == res_b["id"]
