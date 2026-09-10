#!/usr/bin/env python3
"""
Stage 8.6 Diverse Geographic & LULC Dataset Acquisition Pipeline.
Expands BigEarthNet.txt train and validation subsets across multiple European countries:
  - Finland (Boreal forests, lakes)
  - Portugal (Mediterranean scrub, agro-forestry)
  - Austria (Alpine terrain, pastures)
  - Ireland (Atlantic grassland, peatlands)
  - Lithuania (Eastern European mixed agriculture)
  - Serbia (Pannonian arable plains)

Dataset Targets:
- Train: 2,000 unique patches across 5 countries (~15,000 questions)
- Validation: 600 unique patches across 5 countries (~4,500 questions)

Uses mathematically verified patch coordinates:
  col_offset = H_Order * 120
  row_offset = V_Order * 120
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
from concurrent.futures import ThreadPoolExecutor

PARQUET_PATH = "data/bigearthnet/BigEarthNet.txt.parquet"
METADATA_PATH = "data/bigearthnet/metadata_v2.parquet"
RASTER_DIR = "data/bigearthnet/rasters"
MANIFEST_DIR = "data/manifests"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
os.environ["VSI_CACHE"] = "TRUE"


def get_tile_and_coords(patch_id: str):
    parts = patch_id.split("_")
    sensing_date = parts[2][:8]
    date_str = f"{sensing_date[:4]}-{sensing_date[4:6]}-{sensing_date[6:]}"
    mgrs_tile = parts[5][1:] if parts[5].startswith("T") else parts[5]
    h_order = int(parts[6])  # Horizontal -> column
    v_order = int(parts[7])  # Vertical -> row
    return mgrs_tile, date_str, h_order, v_order


def get_sas_token():
    r = requests.get(SAS_URL, timeout=10)
    r.raise_for_status()
    return r.json()["token"]


def read_band_windows(url, windows):
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        VSI_CACHE=True,
        VSI_CACHE_SIZE=100_000_000,
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_MULTIPLEX="YES",
    ):
        with rasterio.open(url) as src:
            return [src.read(1, window=w) for w in windows]


def main():
    os.makedirs(RASTER_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    print("Step 1: Loading BigEarthNet.txt binary annotations...", flush=True)
    pf = pq.ParquetFile(PARQUET_PATH)
    df = pf.read(columns=["ID", "patch_id", "s1_name", "split", "type", "category", "input", "output", "country"]).to_pandas()
    df_bin = df[df["type"] == "binary"].copy()

    def get_scene(pid):
        parts = pid.split("_")
        return f"{parts[5]}_{parts[2][:8]}"

    df_bin["scene"] = df_bin["patch_id"].apply(get_scene)

    # -------------------------------------------------------------
    # Step 2: Multi-Country Deterministic Sampling
    # -------------------------------------------------------------
    # Train: 5 diverse scenes across 5 countries
    train_scene_alloc = [
        ("T35VNK_20180525", 500),  # Finland (already cached)
        ("T29SNC_20180515", 500),  # Portugal (already cached)
        ("T33UWP_20180506", 350),  # Austria (new)
        ("T29UPU_20180421", 350),  # Ireland (new)
        ("T35ULB_20170927", 300),  # Lithuania (new)
    ]
    train_pids = []
    for sc, count in train_scene_alloc:
        sub = df_bin[(df_bin["split"] == "train") & (df_bin["scene"] == sc)]["patch_id"].unique()
        selected = sorted(sub)[:count]
        train_pids.extend(selected)

    df_train = df_bin[df_bin["patch_id"].isin(train_pids)].copy()

    # Val: 5 diverse scenes across 5 countries
    val_scene_alloc = [
        ("T35VNL_20180525", 150),  # Finland (already cached)
        ("T34TEP_20180502", 150),  # Serbia (already cached)
        ("T33UWP_20180506", 100),  # Austria (new)
        ("T29UPU_20180421", 100),  # Ireland (new)
        ("T35ULB_20170927", 100),  # Lithuania (new)
    ]
    val_pids = []
    for sc, count in val_scene_alloc:
        sub = df_bin[(df_bin["split"] == "validation") & (df_bin["scene"] == sc)]["patch_id"].unique()
        selected = sorted(sub)[:count]
        val_pids.extend(selected)

    df_val = df_bin[df_bin["patch_id"].isin(val_pids)].copy()

    print(f"\nTarget Subsets:", flush=True)
    print(f"Train: {len(df_train)} questions across {len(train_pids)} unique patches ({df_train['country'].nunique()} countries)", flush=True)
    print(f"Val:   {len(df_val)} questions across {len(val_pids)} unique patches ({df_val['country'].nunique()} countries)", flush=True)

    # Verify zero leakage
    s_train = set(train_pids)
    s_val = set(val_pids)
    assert len(s_train.intersection(s_val)) == 0, "LEAKAGE: Train and Val overlap!"
    print("Zero-leakage verified: Train and Val patches are 100% disjoint.", flush=True)

    # -------------------------------------------------------------
    # Step 3: Acquire Authentic Sentinel-2 L2A Rasters
    # -------------------------------------------------------------
    all_needed = s_train.union(s_val)
    scene_groups = {}
    for pid in all_needed:
        mgrs, dt, h_ord, v_ord = get_tile_and_coords(pid)
        key = (mgrs, dt)
        if key not in scene_groups:
            scene_groups[key] = []
        scene_groups[key].append((pid, h_ord, v_ord))

    token = get_sas_token()
    acquired_count = 0
    cached_count = 0

    for (mgrs, dt), patch_list in scene_groups.items():
        patch_list = sorted(patch_list, key=lambda x: (x[2], x[1]))
        needed = []
        for pid, h_ord, v_ord in patch_list:
            out_file = os.path.join(RASTER_DIR, f"{pid}_s2_rgb.png")
            if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
                cached_count += 1
            else:
                needed.append((pid, h_ord, v_ord, out_file))

        if not needed:
            print(f"Scene {mgrs} on {dt}: all {len(patch_list)} patches already cached.", flush=True)
            continue

        print(f"\nScene {mgrs} on {dt}: extracting {len(needed)} new patches...", flush=True)
        try:
            resp = requests.post(STAC_URL, json={
                "collections": ["sentinel-2-l2a"],
                "query": {"s2:mgrs_tile": {"eq": mgrs}},
                "datetime": f"{dt}T00:00:00Z/{dt}T23:59:59Z"
            }, timeout=20)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if not features:
                print(f"  ERROR: No STAC item for {mgrs} on {dt}!", flush=True)
                continue

            item = features[0]
            url_r = item["assets"]["B04"]["href"] + "?" + token
            url_g = item["assets"]["B03"]["href"] + "?" + token
            url_b = item["assets"]["B02"]["href"] + "?" + token

            batch_size = 50
            t_scene_start = time.time()

            for b_idx in range(0, len(needed), batch_size):
                sub_batch = needed[b_idx:b_idx + batch_size]
                windows = [Window(col_off=h_ord * 120, row_off=v_ord * 120, width=120, height=120) for _, h_ord, v_ord, _ in sub_batch]

                with ThreadPoolExecutor(max_workers=3) as ex:
                    f4 = ex.submit(read_band_windows, url_r, windows)
                    f3 = ex.submit(read_band_windows, url_g, windows)
                    f2 = ex.submit(read_band_windows, url_b, windows)
                    b4_list = f4.result()
                    b3_list = f3.result()
                    b2_list = f2.result()

                for j, (pid, h_ord, v_ord, out_file) in enumerate(sub_batch):
                    b4 = b4_list[j]
                    b3 = b3_list[j]
                    b2 = b2_list[j]

                    rgb = np.stack([b4, b3, b2], axis=-1).astype(np.float32)
                    rgb_scaled = np.clip(rgb / 3000.0, 0.0, 1.0)
                    rgb_uint8 = (rgb_scaled * 255.0).astype(np.uint8)

                    im = Image.fromarray(rgb_uint8)
                    im.save(out_file)
                    acquired_count += 1

                done = min(b_idx + batch_size, len(needed))
                rate = done / (time.time() - t_scene_start)
                print(f"    Scene {mgrs}: extracted {done}/{len(needed)} patches ({rate:.1f} patches/sec)...", flush=True)

            print(f"  Scene {mgrs} complete in {time.time() - t_scene_start:.1f}s.", flush=True)

        except Exception as e:
            print(f"  ERROR processing scene {mgrs} {dt}: {e}", flush=True)

    print(f"\nAcquisition complete! {acquired_count} new rasters downloaded, {cached_count} cached.", flush=True)

    # -------------------------------------------------------------
    # Step 4: Build & Save Stage 8.6 Manifests
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
                "country": str(row["country"]),
                "local_raster_path": raster_path,
                "source_url": f"PlanetaryComputer/sentinel-2-l2a/reBEN/{pid}",
                "question": str(row["input"]),
                "answer": str(row["output"]).lower().strip(),
                "task": str(row["category"]),
                "bands_used": ["B04", "B03", "B02"]
            })
        print(f"Manifest '{split_name}': {len(records)} records (missing: {missing})", flush=True)
        return records

    train_records = build_manifest(df_train, "train")
    val_records = build_manifest(df_val, "validation")

    with open(os.path.join(MANIFEST_DIR, "bigearthnet_txt_stage8_6_train.json"), "w") as f:
        json.dump(train_records, f, indent=2)
    with open(os.path.join(MANIFEST_DIR, "bigearthnet_txt_stage8_6_val.json"), "w") as f:
        json.dump(val_records, f, indent=2)

    print("\nStage 8.6 diverse manifests successfully saved!", flush=True)


if __name__ == "__main__":
    main()
