"""
SatQuery AI — Real Inference Smoke Test (Prompt 3)
Executes genuine end-to-end model inference using registered specialist adapters:
1. Single-Image VQA (Salesforce/blip-vqa-base)
2. Text-Guided Grounding (google/owlvit-base-patch32)
Saves an annotated visualization artifact with real bounding boxes.
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.transform import from_origin

# Setup path
sys.path.insert(0, os.path.abspath("backend"))

from app.domain.modalities import Modality, ImageFormat
from app.domain.tasks import TaskFamily
from app.schemas.image_asset import ImageAsset
from app.db.asset_repository import asset_repository
from app.registry.model_registry import model_tool_registry
from app.services.models.schemas import VqaRequest, GroundingRequest


def prepare_real_satellite_geotiff(path: str) -> str:
    """Creates a real georeferenced GeoTIFF from actual satellite imagery (public/missions/flood-optical.jpg)."""
    src_jpg = "public/missions/flood-optical.jpg"
    pil_img = Image.open(src_jpg).convert("RGB")
    width, height = pil_img.size
    arr = np.array(pil_img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Georeference in Sundarbans flood delta (EPSG:4326)
    transform = from_origin(88.35, 22.57, 0.0001, 0.0001)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform
    ) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)

    return path


def run_smoke_tests():
    print("==================================================")
    print("SATQUERY AI — PROMPT 3 REAL INFERENCE SMOKE TEST")
    print("==================================================")

    tif_path = os.path.abspath("storage/flood_smoke_test.tif")
    prepare_real_satellite_geotiff(tif_path)
    print(f"[OK] Generated georeferenced GeoTIFF from real satellite scene at: {tif_path}")

    asset_id = "real_satellite_flood_001"
    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height

    asset = ImageAsset(
        id=asset_id,
        filename="flood_smoke_test.tif",
        original_filename="flood_smoke_test.tif",
        storage_path=tif_path,
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=width,
        height=height,
        bands=3,
        crs="EPSG:4326",
        resolution=10.0,
        affine_transform=[0.0001, 0.0, 88.35, 0.0, -0.0001, 22.57],
        validation_status="valid"
    )
    asset_repository.save_asset(asset, tif_path)
    print(f"[OK] Persisted ImageAsset '{asset_id}' ({width}x{height}) into SQLite registry.")

    # 1. Single-Image VQA Smoke Test
    print("\n--- 1. Testing Single-Image VQA Specialist (Salesforce/blip-vqa-base) ---")
    vqa_adapter = model_tool_registry.get_adapter(TaskFamily.SINGLE_VQA)
    assert vqa_adapter is not None, "VqaAdapter failed to resolve from registry."
    print(f"Loaded adapter: {vqa_adapter.model_name}")

    question = "What is visible in this aerial satellite photograph?"
    vqa_req = VqaRequest(asset_id=asset_id, question=question)
    start_vqa = time.time()
    vqa_res = vqa_adapter.predict(vqa_req)
    dur_vqa = time.time() - start_vqa

    print(f"VQA Question: '{question}'")
    print(f"VQA Answer: '{vqa_res.answer}'")
    print(f"VQA Device: {vqa_res.device} | Latency: {vqa_res.inference_latency_ms:.1f}ms (Wall: {dur_vqa*1000:.1f}ms)")
    print(f"Confidence: {vqa_res.confidence} (Uncalibrated per zero-fabrication rules)")

    assert vqa_res.answer and len(vqa_res.answer.strip()) > 0, "VQA returned empty answer!"

    # 2. Text-Guided Grounding Smoke Test
    print("\n--- 2. Testing Text-Guided Grounding Specialist (google/owlvit-base-patch32) ---")
    grounding_adapter = model_tool_registry.get_adapter(TaskFamily.SINGLE_GROUNDING)
    assert grounding_adapter is not None, "GroundingAdapter failed to resolve from registry."
    print(f"Loaded adapter: {grounding_adapter.model_name}")

    query_phrase = "green vegetation"
    grounding_req = GroundingRequest(asset_id=asset_id, queries=[query_phrase, "flooded river"], threshold=0.005)
    start_ground = time.time()
    grounding_res = grounding_adapter.predict(grounding_req)
    dur_ground = time.time() - start_ground

    print(f"Grounding Queries: {[query_phrase, 'flooded river']}")
    print(f"Detected Regions: {len(grounding_res.boxes)}")
    print(f"Grounding Device: {grounding_res.device} | Latency: {grounding_res.inference_latency_ms:.1f}ms (Wall: {dur_ground*1000:.1f}ms)")

    # Render boxes onto visual image for verification artifact
    with rasterio.open(tif_path) as src:
        arr = src.read([1, 2, 3])
        rgb = np.transpose(arr, (1, 2, 0))
        img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    for i, b in enumerate(grounding_res.boxes):
        px = b.pixel_box
        cb = b.canvas_box
        print(f"  Box {i+1}: label='{b.label}' score={b.score:.4f} pixel=[{px[0]}, {px[1]}, {px[2]}, {px[3]}] canvas=[x={cb[0]}%, y={cb[1]}%, w={cb[2]}%, h={cb[3]}%]")
        draw.rectangle([px[0], px[1], px[2], px[3]], outline=(255, 50, 50), width=3)
        draw.text((px[0]+4, px[1]+4), f"{b.label} ({b.score:.3f})", fill=(255, 255, 0))

    artifact_dir = "docs"
    os.makedirs(artifact_dir, exist_ok=True)
    out_artifact = os.path.join(artifact_dir, "grounding_smoke_test_result.png")
    img.save(out_artifact)
    print(f"[OK] Saved grounding verification artifact: {out_artifact}")

    print("\n==================================================")
    print("ALL REAL INFERENCE SMOKE TESTS PASSED SUCCESSFULLY")
    print("==================================================")


if __name__ == "__main__":
    run_smoke_tests()
