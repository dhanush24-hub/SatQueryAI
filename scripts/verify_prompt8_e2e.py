#!/usr/bin/env python3
"""
SatQuery AI — Prompt 8 Real-World End-to-End Validation Script.
Executes and audits all 10 required real-world scenarios (A through J)
under production FastAPI backend with SQLite persistence, report generation,
and strict scientific integrity checks.
"""

import os
import sys
import json
import time
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from fastapi.testclient import TestClient

# Ensure backend is on PYTHONPATH
sys.path.insert(0, os.path.abspath("backend"))

from app.main import app
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.domain.modalities import Modality, ImageFormat
from app.schemas.image_asset import ImageAsset
from app.schemas.analysis import AnalysisRequest
from app.db.asset_repository import asset_repository
from app.db.analysis_repository import analysis_repository
from app.services.storage import storage_service

client = TestClient(app)


def create_geotiff(
    filename: str,
    data: np.ndarray,
    crs_epsg: int = 32636,
    bounds: tuple = (300000, 3300000, 301000, 3301000),
    tags: dict = None
) -> str:
    """Helper to write authentic georeferenced GeoTIFF into storage."""
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


def run_scenario(name: str, description: str, func):
    print(f"\n[{name}] {description}")
    t0 = time.time()
    try:
        func()
        duration = time.time() - t0
        print(f"  -> PASS ({duration:.2f}s)")
        return True
    except Exception as e:
        duration = time.time() - t0
        print(f"  -> FAIL ({duration:.2f}s): {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 75)
    print("SatQuery AI — Prompt 8 / SIH PS 26167 Final End-to-End System Validation")
    print("10 Mandatory Real-World Scenarios (A through J)")
    print("=" * 75)

    results = {}

    # -------------------------------------------------------------------------
    # Scenario A: Single Optical Image VQA & Grounding
    # -------------------------------------------------------------------------
    def test_scenario_a():
        opt_data = np.random.uniform(50, 200, size=(3, 100, 100)).astype(np.uint8)
        f_name = "scenario_a_opt.tif"
        create_geotiff(f_name, opt_data)
        asset = ImageAsset(
            id="scen_a_asset", filename=f_name, original_filename=f_name,
            storage_path=f_name, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=100, height=100, bands=3,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
        )
        asset_repository.save_asset(asset, asset.storage_path)

        res = client.post("/api/analysis", json={"query": "What land cover features are present?", "image_ids": [asset.id]})
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == TaskFamily.SINGLE_VQA
        assert data["confidence"] is None  # Zero fabrication
        assert len(data["execution_summary"]) >= 1

    results["Scenario A (Single Optical Image)"] = run_scenario(
        "SCENARIO A", "Single Optical Image VQA & Grounding", test_scenario_a
    )

    # -------------------------------------------------------------------------
    # Scenario B: Supported Temporal Optical Pair with Change
    # -------------------------------------------------------------------------
    def test_scenario_b():
        np.random.seed(101)
        base = np.random.randint(40, 180, (3, 256, 256), dtype=np.uint8)
        t1_data = base.copy()
        t2_data = base.copy()
        # Add new building cluster in T2
        t2_data[:, 40:90, 40:90] = 230

        f_t1, f_t2 = "scenario_b_t1.tif", "scenario_b_t2.tif"
        create_geotiff(f_t1, t1_data)
        create_geotiff(f_t2, t2_data)

        a1 = ImageAsset(
            id="scen_b_t1", filename=f_t1, original_filename=f_t1,
            storage_path=f_t1, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=256, height=256, bands=3,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], acquisition_time="2020-01-01T00:00:00Z", validation_status="valid"
        )
        a2 = ImageAsset(
            id="scen_b_t2", filename=f_t2, original_filename=f_t2,
            storage_path=f_t2, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=256, height=256, bands=3,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], acquisition_time="2022-01-01T00:00:00Z", validation_status="valid"
        )
        asset_repository.save_asset(a1, a1.storage_path)
        asset_repository.save_asset(a2, a2.storage_path)

        res = client.post("/api/analysis", json={"query": "What changed between these two dates?", "image_ids": [a1.id, a2.id]})
        assert res.status_code == 200
        data = res.json()
        assert data["task"] in [TaskFamily.TEMPORAL_CHANGE, TaskFamily.TEMPORAL_CHANGE_VQA]
        assert data["confidence"] is None
        assert len(data["findings"]) > 0
        assert data["downloadable_artifacts"]["report_pdf_url"] is not None

    results["Scenario B (Temporal Pair with Change)"] = run_scenario(
        "SCENARIO B", "Supported Temporal Pair with Construction Change", test_scenario_b
    )

    # -------------------------------------------------------------------------
    # Scenario C: Temporal Optical Pair with No Meaningful Change
    # -------------------------------------------------------------------------
    def test_scenario_c():
        np.random.seed(102)
        base_data = np.random.randint(80, 160, (3, 100, 100), dtype=np.uint8)
        f_t1, f_t2 = "scenario_c_t1.tif", "scenario_c_t2.tif"
        create_geotiff(f_t1, base_data)
        create_geotiff(f_t2, base_data)  # Identical baseline

        a1 = ImageAsset(
            id="scen_c_t1", filename=f_t1, original_filename=f_t1,
            storage_path=f_t1, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=100, height=100, bands=3,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], acquisition_time="2021-01-01T00:00:00Z", validation_status="valid"
        )
        a2 = ImageAsset(
            id="scen_c_t2", filename=f_t2, original_filename=f_t2,
            storage_path=f_t2, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=100, height=100, bands=3,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], acquisition_time="2022-01-01T00:00:00Z", validation_status="valid"
        )
        asset_repository.save_asset(a1, a1.storage_path)
        asset_repository.save_asset(a2, a2.storage_path)

        res = client.post("/api/analysis", json={"query": "Did any construction change occur?", "image_ids": [a1.id, a2.id]})
        assert res.status_code == 200
        data = res.json()
        assert "no significant" in data["answer"].lower() or "no_change_detected" in data["answer"].lower() or len(data["findings"]) == 0
        assert data["confidence"] is None

    results["Scenario C (Identical Baseline - No Change)"] = run_scenario(
        "SCENARIO C", "Temporal Optical Pair with Zero Change", test_scenario_c
    )

    # -------------------------------------------------------------------------
    # Scenario D: Authentic Optical + SAR Pair
    # -------------------------------------------------------------------------
    def test_scenario_d():
        opt_data = np.ones((3, 60, 60), dtype=np.uint8) * 140
        opt_data[:, 30:, 30:] = 20  # Water
        sar_data = np.ones((1, 60, 60), dtype=np.float32) * (-8.0)
        sar_data[0, 30:, 30:] = -22.0  # Specular water

        f_opt, f_sar = "scenario_d_opt.tif", "scenario_d_sar.tif"
        create_geotiff(f_opt, opt_data)
        create_geotiff(f_sar, sar_data, tags={"SENSOR": "SENTINEL-1", "SIGMA0_DB": "CALIBRATED"})

        a_opt = ImageAsset(
            id="scen_d_opt", filename=f_opt, original_filename=f_opt,
            storage_path=f_opt, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=60, height=60, bands=3,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
        )
        a_sar = ImageAsset(
            id="scen_d_sar", filename=f_sar, original_filename=f_sar,
            storage_path=f_sar, format=ImageFormat.GEOTIFF,
            modality=Modality.SAR, width=60, height=60, bands=1,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
        )
        asset_repository.save_asset(a_opt, a_opt.storage_path)
        asset_repository.save_asset(a_sar, a_sar.storage_path)

        res = client.post("/api/analysis", json={"query": "Extract water extent across optical and SAR", "image_ids": [a_opt.id, a_sar.id]})
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == TaskFamily.OPTICAL_SAR_ANALYSIS
        assert data["confidence"] is None
        assert any(f["type"] == "cross_modal_agreement" for f in data["findings"])
        assert data["model_provenance"]["model_name"] == "SatQuery-OpticalSAR-FusionAdapter-v1"

    results["Scenario D (Authentic Optical+SAR Pair)"] = run_scenario(
        "SCENARIO D", "Authentic Optical+SAR Cross-Modal Fusion", test_scenario_d
    )

    # -------------------------------------------------------------------------
    # Scenario E: Optical/SAR Disagreement
    # -------------------------------------------------------------------------
    def test_scenario_e():
        opt_data = np.ones((3, 60, 60), dtype=np.uint8) * 140
        opt_data[:, 20:50, 20:50] = 20  # optical water candidate
        sar_data = np.ones((1, 60, 60), dtype=np.float32) * (-10.0)
        sar_data[0, 20:50, 20:50] = 2.0  # high backscatter (urban structure / dry rough soil)

        f_opt, f_sar = "scenario_e_opt.tif", "scenario_e_sar.tif"
        create_geotiff(f_opt, opt_data)
        create_geotiff(f_sar, sar_data, tags={"SENSOR": "SENTINEL-1", "SIGMA0_DB": "CALIBRATED"})

        a_opt = ImageAsset(
            id="scen_e_opt", filename=f_opt, original_filename=f_opt,
            storage_path=f_opt, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=60, height=60, bands=3,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
        )
        a_sar = ImageAsset(
            id="scen_e_sar", filename=f_sar, original_filename=f_sar,
            storage_path=f_sar, format=ImageFormat.GEOTIFF,
            modality=Modality.SAR, width=60, height=60, bands=1,
            crs="EPSG:32636", resolution=10.0, affine_transform=[10, 0, 300000, 0, -10, 3301000],
            geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
        )
        asset_repository.save_asset(a_opt, a_opt.storage_path)
        asset_repository.save_asset(a_sar, a_sar.storage_path)

        res = client.post("/api/analysis", json={"query": "Assess water inundation from optical and radar", "image_ids": [a_opt.id, a_sar.id]})
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == TaskFamily.OPTICAL_SAR_ANALYSIS
        assert any(f["type"] == "cross_modal_disagreement" for f in data["findings"])
        assert "divergence" in data["answer"].lower() or "disagree" in str(data["warnings"]).lower() or data["evidence_state"] in ["CONFLICTING", "SUPPORTED_WITH_WARNINGS", "WEAK"]

    results["Scenario E (Optical/SAR Disagreement)"] = run_scenario(
        "SCENARIO E", "Sensor Divergence Surfaced (Optical Shadow vs SAR Structure)", test_scenario_e
    )

    # -------------------------------------------------------------------------
    # Scenario F: Poor Cross-Modal Registration
    # -------------------------------------------------------------------------
    def test_scenario_f():
        opt_data = np.random.uniform(50, 200, size=(3, 50, 50)).astype(np.uint8)
        sar_data = np.random.uniform(-30, 0, size=(1, 50, 50)).astype(np.float32)

        f_opt, f_sar = "scenario_f_opt.tif", "scenario_f_sar.tif"
        create_geotiff(f_opt, opt_data)
        create_geotiff(f_sar, sar_data, crs_epsg=32637, bounds=(500000, 4000000, 501000, 4001000))

        a_opt = ImageAsset(
            id="scen_f_opt", filename=f_opt, original_filename=f_opt,
            storage_path=f_opt, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=50, height=50, bands=3,
            crs="EPSG:32636", geographic_bbox=[31.0, 30.0, 31.01, 30.01], validation_status="valid"
        )
        a_sar = ImageAsset(
            id="scen_f_sar", filename=f_sar, original_filename=f_sar,
            storage_path=f_sar, format=ImageFormat.GEOTIFF,
            modality=Modality.SAR, width=50, height=50, bands=1,
            crs="EPSG:32637", geographic_bbox=[45.0, 36.0, 45.01, 36.01], validation_status="valid"
        )
        asset_repository.save_asset(a_opt, a_opt.storage_path)
        asset_repository.save_asset(a_sar, a_sar.storage_path)

        res = client.post("/api/analysis", json={"query": "Fuse misaligned optical and SAR", "image_ids": [a_opt.id, a_sar.id]})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == AnalysisStatus.VALIDATION_FAILED
        assert "incompatible" in data["answer"].lower()

    results["Scenario F (Poor Cross-Modal Registration)"] = run_scenario(
        "SCENARIO F", "Mismatched CRS and Footprints Rejected at Gate", test_scenario_f
    )

    # -------------------------------------------------------------------------
    # Scenario G: Out-of-Domain Imagery
    # -------------------------------------------------------------------------
    def test_scenario_g():
        # Tiny 8x8 image (violates operational domain)
        tiny_data = np.zeros((3, 8, 8), dtype=np.uint8)
        f_tiny = "scenario_g_tiny.tif"
        create_geotiff(f_tiny, tiny_data)

        asset = ImageAsset(
            id="scen_g_tiny", filename=f_tiny, original_filename=f_tiny,
            storage_path=f_tiny, format=ImageFormat.GEOTIFF,
            modality=Modality.OPTICAL, width=8, height=8, bands=3,
            validation_status="valid"
        )
        asset_repository.save_asset(asset, asset.storage_path)

        res = client.post("/api/analysis", json={"query": "Describe tiny image", "image_ids": [asset.id]})
        assert res.status_code == 200
        data = res.json()
        assert data["evidence_state"] in ["OUT_OF_DOMAIN", "INSUFFICIENT_EVIDENCE", "UNAVAILABLE"] or any("domain" in str(w).lower() or "too small" in str(w).lower() for w in data["warnings"])

    results["Scenario G (Out-of-Domain Imagery)"] = run_scenario(
        "SCENARIO G", "Out-of-Domain / Micro-Dimension Input Gating", test_scenario_g
    )

    # -------------------------------------------------------------------------
    # Scenario H: Unsupported Semantic Query
    # -------------------------------------------------------------------------
    def test_scenario_h():
        t1_data = np.random.randint(40, 180, (3, 100, 100), dtype=np.uint8)
        t2_data = t1_data.copy()
        f_h1, f_h2 = "scenario_h_t1.tif", "scenario_h_t2.tif"
        create_geotiff(f_h1, t1_data)
        create_geotiff(f_h2, t2_data)

        t1 = ImageAsset(
            id="scen_h_t1", filename=f_h1, original_filename=f_h1,
            storage_path=f_h1, format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL,
            width=100, height=100, bands=3, crs="EPSG:32636", geographic_bbox=[31.0, 30.0, 31.01, 30.01],
            validation_status="benchmark"
        )
        t2 = ImageAsset(
            id="scen_h_t2", filename=f_h2, original_filename=f_h2,
            storage_path=f_h2, format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL,
            width=100, height=100, bands=3, crs="EPSG:32636", geographic_bbox=[31.0, 30.0, 31.01, 30.01],
            validation_status="benchmark"
        )
        asset_repository.save_asset(t1, f_h1)
        asset_repository.save_asset(t2, f_h2)

        res = client.post("/api/analysis", json={"query": "Was forest converted to buildings here?", "image_ids": [t1.id, t2.id]})
        assert res.status_code == 200
        data = res.json()
        assert "cannot reliably determine the land-cover transition type" in data["answer"]
        assert any("transition" in w.lower() for w in data["warnings"])

    results["Scenario H (Unsupported Semantic Transition Safeguard)"] = run_scenario(
        "SCENARIO H", "Semantic Safeguard for Conversion Questions", test_scenario_h
    )

    # -------------------------------------------------------------------------
    # Scenario I: Missing Metadata (Unreferenced Pixel Grid)
    # -------------------------------------------------------------------------
    def test_scenario_i():
        from PIL import Image as PILImage
        img_arr = np.random.randint(40, 180, (100, 100, 3), dtype=np.uint8)
        img_i = PILImage.fromarray(img_arr)
        f_i1, f_i2 = "scenario_i_t1.png", "scenario_i_t2.png"
        img_i.save(storage_service.get_full_path(f_i1))
        img_i.save(storage_service.get_full_path(f_i2))

        t1_unref = ImageAsset(
            id="scen_i_t1", filename=f_i1, original_filename=f_i1,
            storage_path=f_i1, format=ImageFormat.PNG, modality=Modality.OPTICAL,
            width=100, height=100, bands=3, crs=None, geographic_bbox=None,
            validation_status="benchmark"
        )
        t2_unref = ImageAsset(
            id="scen_i_t2", filename=f_i2, original_filename=f_i2,
            storage_path=f_i2, format=ImageFormat.PNG, modality=Modality.OPTICAL,
            width=100, height=100, bands=3, crs=None, geographic_bbox=None,
            validation_status="benchmark"
        )
        asset_repository.save_asset(t1_unref, f_i1)
        asset_repository.save_asset(t2_unref, f_i2)

        res = client.post("/api/analysis", json={"query": "What changed between these images?", "image_ids": [t1_unref.id, t2_unref.id]})
        assert res.status_code == 200
        data = res.json()
        assert "physical area omitted" in data["answer"].lower() or data["downloadable_artifacts"]["geojson_url"] is None
        assert data["confidence"] is None

    results["Scenario I (Missing Metadata - Unreferenced Area Suppression)"] = run_scenario(
        "SCENARIO I", "Physical Area and GeoJSON Suppressed Without CRS", test_scenario_i
    )

    # -------------------------------------------------------------------------
    # Scenario J: Reopen Persisted Historical Result
    # -------------------------------------------------------------------------
    def test_scenario_j():
        # Retrieve recent completed analysis record from database
        history = analysis_repository.list_analyses(limit=10)
        assert len(history) > 0
        target_rec = next((r for r in history if r.status in [AnalysisStatus.COMPLETED, AnalysisStatus.COMPLETED_WITH_WARNINGS]), history[0])
        target_id = target_rec.id

        res = client.get(f"/api/analysis/{target_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == target_id
        assert data["answer"] is not None
        assert data["status"] in [AnalysisStatus.COMPLETED, AnalysisStatus.COMPLETED_WITH_WARNINGS, AnalysisStatus.VALIDATION_FAILED, AnalysisStatus.INSUFFICIENT_EVIDENCE]
        assert data["confidence"] is None
        assert data["downloadable_artifacts"] is not None

    results["Scenario J (Reopen Persisted Historical Result)"] = run_scenario(
        "SCENARIO J", "SQLite History Retrieval & Reconstruction", test_scenario_j
    )

    print("\n" + "=" * 75)
    print("E2E SCENARIO SUMMARY MATRIX:")
    print("-" * 75)
    all_passed = True
    for sc_name, passed in results.items():
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{sc_name:<60} : {status_str}")
    print("=" * 75)

    if all_passed:
        print("ALL 10 REAL-WORLD E2E SCENARIOS PASSED WITH ZERO FABRICATION!")
        return 0
    else:
        print("SOME SCENARIOS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
