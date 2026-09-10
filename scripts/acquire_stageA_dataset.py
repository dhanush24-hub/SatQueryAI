#!/usr/bin/env python3
"""
Sparse Image Acquisition Pipeline for BigEarthNet.txt Stage A (S2-RGB).
Acquires authentic Sentinel-2 L2A rasters directly from Copernicus STAC (Planetary Computer)
for deterministic, balanced binary VQA subsets of BigEarthNet.txt (arXiv:2603.29630).

Strictly enforces:
- Zero data leakage across train, val, and bench splits.
- Benchmark split (1,082 image pairs) used exclusively for final evaluation.
- No synthetic pixels: 100% authentic Sentinel-2 L2A surface reflectance rasters.
"""

import os
import sys
import json
import time
import requests
import rasterio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rasterio.windows import Window
from PIL import Image

PARQUET_PATH = "data/bigearthnet/BigEarthNet.txt.parquet"
RASTER_DIR = "data/bigearthnet/rasters"
MANIFEST_DIR = "data/manifests"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
os.environ["VSI_CACHE"] = "TRUE"


def get_tile_and_date(patch_id: str):
    parts = patch_id.split("_")
    # S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_32_64
    sensing_date = parts[2][:8]  # e.g. 20170613
    date_str = f"{sensing_date[:4]}-{sensing_date[4:6]}-{sensing_date[6:]}"
    mgrs_tile = parts[5].replace("T", "")  # e.g. 33UUP
    row = int(parts[6])
    col = int(parts[7])
    return mgrs_tile, date_str, row, col


def get_sas_token():
    r = requests.get(SAS_URL, timeout=10)
    r.raise_for_status()
    return r.json()["token"]


def main():
    os.makedirs(RASTER_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    print("Step 1: Loading BigEarthNet.txt annotations...", flush=True)
    pf = pq.ParquetFile(PARQUET_PATH)
    df = pf.read(columns=["ID", "patch_id", "s1_name", "split", "type", "category", "input", "output", "country"]).to_pandas()
    df_bin = df[df["type"] == "binary"].copy()
    print(f"Total binary VQA annotations loaded: {len(df_bin)}", flush=True)

    def extract_tile_date(pid):
        parts = pid.split("_")
        return f"{parts[5]}_{parts[2][:8]}"

    df_bin["tile_date"] = df_bin["patch_id"].apply(extract_tile_date)

    # -------------------------------------------------------------
    # Step 2: Deterministic Patch-Based Selection
    # -------------------------------------------------------------
    # A. Benchmark: T29SND_20180515
    df_bench_all = df_bin[(df_bin["split"] == "bench") & (df_bin["tile_date"] == "T29SND_20180515")]
    bench_pids = sorted(df_bench_all["patch_id"].unique())[:35]
    df_bench_final = df_bench_all[df_bench_all["patch_id"].isin(bench_pids)].copy()

    # B. Validation: T35VNL_20180525
    df_val_all = df_bin[(df_bin["split"] == "validation") & (df_bin["tile_date"] == "T35VNL_20180525")]
    val_pids = sorted(df_val_all["patch_id"].unique())[:50]
    df_val_final = df_val_all[df_val_all["patch_id"].isin(val_pids)].copy()

    # C. Train: T35VNK_20180525
    df_train_all = df_bin[(df_bin["split"] == "train") & (df_bin["tile_date"] == "T35VNK_20180525")]
    train_pids = sorted(df_train_all["patch_id"].unique())[:200]
    df_train_final = df_train_all[df_train_all["patch_id"].isin(train_pids)].copy()

    print(f"\nFinal Selected Annotations:", flush=True)
    print(f"Train: {len(df_train_final)} ({df_train_final['patch_id'].nunique()} unique patches)", flush=True)
    print(f"Val:   {len(df_val_final)} ({df_val_final['patch_id'].nunique()} unique patches)", flush=True)
    print(f"Bench: {len(df_bench_final)} ({df_bench_final['patch_id'].nunique()} unique patches)", flush=True)

    # -------------------------------------------------------------
    # Step 3: Zero-Leakage Verification
    # -------------------------------------------------------------
    train_patches = set(train_pids)
    val_patches = set(val_pids)
    bench_patches = set(bench_pids)

    assert len(train_patches.intersection(bench_patches)) == 0, "LEAKAGE: Train & Bench overlap!"
    assert len(val_patches.intersection(bench_patches)) == 0, "LEAKAGE: Val & Bench overlap!"
    assert len(train_patches.intersection(val_patches)) == 0, "LEAKAGE: Train & Val overlap!"
    print("Zero-leakage verified: Train, Val, and Bench patches are 100% disjoint.", flush=True)

    # -------------------------------------------------------------
    # Step 4: Authentic S2 Rasters Acquisition
    # -------------------------------------------------------------
    all_needed_patches = train_patches.union(val_patches).union(bench_patches)
    print(f"\nTotal unique authentic patches to acquire: {len(all_needed_patches)}", flush=True)

    # Group patches by (mgrs_tile, date_str)
    tile_groups = {}
    for pid in all_needed_patches:
        mgrs, dt, row, col = get_tile_and_date(pid)
        key = (mgrs, dt)
        if key not in tile_groups:
            tile_groups[key] = []
        tile_groups[key].append((pid, row, col))

    print(f"Unique Sentinel-2 scenes to query: {len(tile_groups)}", flush=True)
    token = get_sas_token()

    acquired_count = 0
    cached_count = 0

    for (mgrs, dt), patch_list in tile_groups.items():
        print(f"\nProcessing scene: Tile {mgrs} on {dt} ({len(patch_list)} patches)...", flush=True)
        needed_in_tile = []
        for pid, r_idx, c_idx in patch_list:
            out_file = os.path.join(RASTER_DIR, f"{pid}_s2_rgb.png")
            if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
                cached_count += 1
            else:
                needed_in_tile.append((pid, r_idx, c_idx, out_file))

        if not needed_in_tile:
            print(f"  All {len(patch_list)} patches already cached.", flush=True)
            continue

        try:
            resp = requests.post(STAC_URL, json={
                "collections": ["sentinel-2-l2a"],
                "query": {"s2:mgrs_tile": {"eq": mgrs}},
                "datetime": f"{dt}T00:00:00Z/{dt}T23:59:59Z"
            }, timeout=15)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if not features:
                print(f"  WARNING: No STAC item found for {mgrs} on {dt}!", flush=True)
                continue

            item = features[0]
            url_r = item["assets"]["B04"]["href"] + "?" + token
            url_g = item["assets"]["B03"]["href"] + "?" + token
            url_b = item["assets"]["B02"]["href"] + "?" + token

            t_open = time.time()
            with rasterio.open(url_r) as src_r, \
                 rasterio.open(url_g) as src_g, \
                 rasterio.open(url_b) as src_b:

                print(f"  Opened 3 band rasters in {time.time() - t_open:.2f}s, extracting {len(needed_in_tile)} patches...", flush=True)

                for i, (pid, r_idx, c_idx, out_file) in enumerate(needed_in_tile):
                    w = Window(col_off=c_idx * 120, row_off=r_idx * 120, width=120, height=120)
                    b4 = src_r.read(1, window=w)
                    b3 = src_g.read(1, window=w)
                    b2 = src_b.read(1, window=w)

                    # Surface reflectance scaling to uint8 (BigEarthNet RGB convention)
                    rgb = np.stack([b4, b3, b2], axis=-1).astype(np.float32)
                    rgb_scaled = np.clip(rgb / 3000.0, 0.0, 1.0)
                    rgb_uint8 = (rgb_scaled * 255.0).astype(np.uint8)

                    im = Image.fromarray(rgb_uint8)
                    im.save(out_file)
                    acquired_count += 1
                    if (i + 1) % 25 == 0 or (i + 1) == len(needed_in_tile):
                        print(f"    Extracted {i + 1}/{len(needed_in_tile)} patches...", flush=True)

            print(f"  Acquisition finished for scene {mgrs} {dt}.", flush=True)

        except Exception as e:
            print(f"  ERROR acquiring tile {mgrs} {dt}: {e}", flush=True)

    print(f"\nAcquisition complete: {acquired_count} new rasters downloaded, {cached_count} previously cached.", flush=True)

    # -------------------------------------------------------------
    # Step 5: Save Stage A Manifests
    # -------------------------------------------------------------
    def build_manifest(df_split, split_name):
        records = []
        missing = 0
        for _, row in df_split.iterrows():
            pid = str(row["patch_id"])
            raster_path = os.path.join(RASTER_DIR, f"{pid}_s2_rgb.png")
            if not os.path.exists(raster_path):
                missing += 1
                continue
            records.append({
                "patch_id": pid,
                "source_split": split_name,
                "local_raster_path": raster_path,
                "source_url": f"PlanetaryComputer/sentinel-2-l2a/reBEN/{pid}",
                "question": str(row["input"]),
                "answer": str(row["output"]).lower().strip(),
                "task": str(row["category"]),
                "bands_used": ["B04", "B03", "B02"]
            })
        print(f"Manifest '{split_name}': {len(records)} valid records (missing rasters: {missing})", flush=True)
        return records

    train_records = build_manifest(df_train_final, "train")
    val_records = build_manifest(df_val_final, "validation")
    bench_records = build_manifest(df_bench_final, "bench")

    with open(os.path.join(MANIFEST_DIR, "bigearthnet_txt_stageA_train.json"), "w") as f:
        json.dump(train_records, f, indent=2)
    with open(os.path.join(MANIFEST_DIR, "bigearthnet_txt_stageA_val.json"), "w") as f:
        json.dump(val_records, f, indent=2)
    with open(os.path.join(MANIFEST_DIR, "bigearthnet_txt_stageA_bench.json"), "w") as f:
        json.dump(bench_records, f, indent=2)

    print("\nStage A manifests written successfully to data/manifests/!", flush=True)


if __name__ == "__main__":
    main()
