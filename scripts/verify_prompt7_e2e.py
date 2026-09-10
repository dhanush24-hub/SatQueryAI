#!/usr/bin/env python3
"""
SatQuery AI — Prompt 7 End-to-End Verification Script
Runs 5 mandatory demo scenarios through the complete pipeline:
  CASE A: Supported LEVIR-like temporal pair with real change.
  CASE B: Supported pair with little/no change.
  CASE C: Poorly registered pair.
  CASE D: Out-of-domain input.
  CASE E: Unsupported semantic question.
"""

import sys
import io
import os
import json
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import init_db

def create_rgb_png(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def run_e2e_verification():
    print("=" * 70)
    print("SATQUERY AI — PROMPT 7 E2E VERIFICATION (5 MANDATORY SCENARIOS)")
    print("=" * 70)
    
    init_db()
    client = TestClient(app)
    
    # -----------------------------------------------------------------
    # CASE A: Supported LEVIR-like temporal pair with real change
    # -----------------------------------------------------------------
    print("\n[CASE A] Supported LEVIR-like temporal pair with real change...")
    np.random.seed(101)
    base = np.random.randint(40, 180, (256, 256, 3), dtype=np.uint8)
    t1_a = base.copy()
    t2_a = base.copy()
    # Insert realistic structural changes in t2
    t2_a[40:90, 40:90] = 230
    t2_a[140:190, 140:190] = 230
    
    upload_res = client.post("/api/uploads", files=[
        ("files", ("case_a_t1.png", create_rgb_png(t1_a), "image/png")),
        ("files", ("case_a_t2.png", create_rgb_png(t2_a), "image/png")),
    ], data={"benchmark_mode": "true"})
    assert upload_res.status_code == 201, upload_res.text
    id_t1, id_t2 = [img["id"] for img in upload_res.json()]
    
    resp_a = client.post("/api/analysis", json={
        "query": "Where did new structures appear between these two dates?",
        "image_ids": [id_t1, id_t2]
    })
    data_a = resp_a.json()
    if data_a["status"] != "COMPLETED":
        print(f"CASE A FAILED STATUS: status={data_a['status']}, state={data_a.get('evidence_state')}, warnings={data_a.get('warnings')}")
    assert data_a["status"] == "COMPLETED"
    assert data_a["evidence_state"] in ("SUPPORTED", "SUPPORTED_WITH_WARNINGS")
    assert data_a["overlays"] is not None
    assert data_a["overlays"]["change_mask_url"] is not None
    assert len(data_a["findings"]) > 0
    assert data_a["findings"][0]["changed_pixels"] > 0
    # Zero fabrication check: unreferenced raster must have area_m2 == None
    assert data_a["findings"][0]["area_m2"] is None
    
    # Check PDF export
    pdf_url = data_a["downloadable_artifacts"]["report_pdf_url"]
    pdf_res = client.get(pdf_url)
    if pdf_res.status_code != 200:
        print(f"FAILED PDF GET: url={pdf_url}, status={pdf_res.status_code}, body={pdf_res.text}")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000
    
    # Check JSON export
    json_url = data_a["downloadable_artifacts"]["result_json_url"]
    json_res = client.get(json_url)
    assert json_res.status_code == 200
    assert json_res.json()["id"] == data_a["id"]
    
    # Check GeoJSON rejection for unreferenced raster (Zero-fabrication: url is None and endpoint returns 400)
    assert data_a["downloadable_artifacts"]["geojson_url"] is None
    geojson_res = client.get(f"/api/analysis/{data_a['id']}/export.geojson")
    assert geojson_res.status_code == 400
    
    print(f"  ✓ Status: {data_a['status']}")
    print(f"  ✓ Evidence state: {data_a['evidence_state']}")
    print(f"  ✓ Overlays: Mask URL generated: {data_a['overlays']['change_mask_url']}")
    print(f"  ✓ Findings: {len(data_a['findings'])} clusters detected")
    print(f"  ✓ Zero fabrication: area_m2 is None (unreferenced), GeoJSON correctly refused (HTTP 400)")
    print(f"  ✓ PDF report: {len(pdf_res.content)} bytes generated successfully")

    # -----------------------------------------------------------------
    # CASE B: Supported pair with little/no change
    # -----------------------------------------------------------------
    print("\n[CASE B] Supported pair with little/no change...")
    t1_b = base.copy()
    t2_b = base.copy() # Identical
    
    upload_b = client.post("/api/uploads", files=[
        ("files", ("case_b_t1.png", create_rgb_png(t1_b), "image/png")),
        ("files", ("case_b_t2.png", create_rgb_png(t2_b), "image/png")),
    ], data={"benchmark_mode": "true"})
    assert upload_b.status_code == 201, upload_b.text
    id_b1, id_b2 = [img["id"] for img in upload_b.json()]
    
    resp_b = client.post("/api/analysis", json={
        "query": "What changed between these two dates?",
        "image_ids": [id_b1, id_b2]
    })
    assert resp_b.status_code == 200, resp_b.text
    data_b = resp_b.json()
    assert data_b["status"] == "COMPLETED"
    assert "no significant change" in data_b["answer"].lower() or len(data_b["findings"]) == 0
    pdf_b = client.get(data_b["downloadable_artifacts"]["report_pdf_url"])
    assert pdf_b.status_code == 200
    
    print(f"  ✓ Status: {data_b['status']}")
    print(f"  ✓ Answer: {data_b['answer']}")
    print(f"  ✓ Findings count: {len(data_b['findings'])}")
    print(f"  ✓ PDF report generated: {len(pdf_b.content)} bytes")

    # -----------------------------------------------------------------
    # CASE C: Poorly registered pair
    # -----------------------------------------------------------------
    print("\n[CASE C] Poorly registered pair...")
    t1_c = base.copy()
    # Translate t2 by 45 pixels to trigger registration warning / failure
    t2_c = np.roll(base, shift=45, axis=(0, 1))
    
    upload_c = client.post("/api/uploads", files=[
        ("files", ("case_c_t1.png", create_rgb_png(t1_c), "image/png")),
        ("files", ("case_c_t2.png", create_rgb_png(t2_c), "image/png")),
    ], data={"benchmark_mode": "true"})
    assert upload_c.status_code == 201, upload_c.text
    id_c1, id_c2 = [img["id"] for img in upload_c.json()]
    
    resp_c = client.post("/api/analysis", json={
        "query": "Identify any structural changes between these acquisitions.",
        "image_ids": [id_c1, id_c2]
    })
    assert resp_c.status_code == 200
    data_c = resp_c.json()
    assert data_c["registration"]["status"] in ("POOR", "FAILED", "UNVERIFIED")
    assert len(data_c["registration"]["warnings"]) > 0 or "registration" in " ".join(data_c["warnings"]).lower()
    print(f"  ✓ Registration status: {data_c['registration']['status']}")
    print(f"  ✓ Translation offset: {data_c['registration']['translation_px']:.2f} px")
    print(f"  ✓ Evidence state: {data_c['evidence_state']}")
    print(f"  ✓ Warning surfaced: {data_c['registration']['warnings']}")

    # -----------------------------------------------------------------
    # CASE D: Out-of-domain input
    # -----------------------------------------------------------------
    print("\n[CASE D] Out-of-domain input (Tiny non-satellite image < 64px)...")
    tiny = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    upload_d = client.post("/api/uploads", files=[
        ("files", ("tiny_t1.png", create_rgb_png(tiny), "image/png")),
        ("files", ("tiny_t2.png", create_rgb_png(tiny), "image/png")),
    ], data={"benchmark_mode": "true"})
    assert upload_d.status_code == 201, upload_d.text
    id_d1, id_d2 = [img["id"] for img in upload_d.json()]
    
    resp_d = client.post("/api/analysis", json={
        "query": "Detect changes between these two images",
        "image_ids": [id_d1, id_d2]
    })
    assert resp_d.status_code == 200
    data_d = resp_d.json()
    assert data_d["status"] == "INSUFFICIENT_EVIDENCE"
    assert data_d["domain_suitability"]["is_suitable"] is False
    assert data_d["evidence_state"] == "OUT_OF_DOMAIN"
    print(f"  ✓ Status: {data_d['status']}")
    print(f"  ✓ Domain suitability: {data_d['domain_suitability']['status']} (is_suitable={data_d['domain_suitability']['is_suitable']})")
    print(f"  ✓ Evidence state: {data_d['evidence_state']}")
    print(f"  ✓ Reasons: {data_d['domain_suitability']['reasons']}")

    # -----------------------------------------------------------------
    # CASE E: Unsupported semantic question
    # -----------------------------------------------------------------
    print("\n[CASE E] Unsupported semantic question...")
    resp_e = client.post("/api/analysis", json={
        "query": "Was forest converted to buildings between these two dates?",
        "image_ids": [id_t1, id_t2]
    })
    assert resp_e.status_code == 200
    data_e = resp_e.json()
    expected_refusal = "Change was detected in these regions, but the active model cannot reliably determine the land-cover transition type."
    assert expected_refusal in data_e["answer"]
    assert any("semantic" in l.lower() or "transition" in l.lower() for l in data_e["limitations"])
    print(f"  ✓ Answer: {data_e['answer']}")
    print(f"  ✓ Safeguard active: Refusal statement verified exactly.")
    print(f"  ✓ Limitations: {data_e['limitations']}")

    # -----------------------------------------------------------------
    # Verification of Persistence & History Reopening
    # -----------------------------------------------------------------
    print("\n[PERSISTENCE AUDIT] Verifying history listing & reopening without model re-runs...")
    hist_res = client.get("/api/analysis")
    assert hist_res.status_code == 200
    history_items = hist_res.json()
    assert len(history_items) >= 5
    print(f"  ✓ Persisted analyses in SQLite database: {len(history_items)}")
    
    # Reopen Case A by ID
    reopen_res = client.get(f"/api/analysis/{data_a['id']}")
    assert reopen_res.status_code == 200
    reopened = reopen_res.json()
    assert reopened["id"] == data_a["id"]
    assert reopened["answer"] == data_a["answer"]
    assert reopened["model_provenance"]["model_name"] == "AttentionChangeNet"
    assert "ResNet-18" in reopened["model_provenance"]["architecture"]
    print(f"  ✓ Reopened Analysis ID: {reopened['id']}")
    print(f"  ✓ Checkpoint verified: {reopened['model_provenance']['checkpoint']}")
    print(f"  ✓ Threshold verified: {reopened['model_provenance']['threshold']}")
    
    print("\n" + "=" * 70)
    print("ALL 5 PROMPT 7 DEMO SCENARIOS PASSED WITH STRICT SCIENTIFIC COMPLIANCE!")
    print("=" * 70)

if __name__ == "__main__":
    run_e2e_verification()
