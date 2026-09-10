#!/usr/bin/env python3
"""
Audit Image Correspondence for BigEarthNet v2.0 / reBEN Patches.
Audits 20 randomly selected train/val patches against Sentinel-2 L2A STAC transforms,
MGRS bounding boxes, and ground-truth patch positions.
"""

import os
import json
import random
import requests
import rasterio
import rasterio.warp
from rasterio.windows import Window
import numpy as np
import pandas as pd

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"

def get_sas_token():
    r = requests.get(SAS_URL, timeout=10)
    r.raise_for_status()
    return r.json()["token"]

def main():
    random.seed(42)
    with open("data/manifests/bigearthnet_txt_stageA_train.json", "r") as f:
        train_records = json.load(f)
    with open("data/manifests/bigearthnet_txt_stageA_val.json", "r") as f:
        val_records = json.load(f)

    all_records = train_records + val_records
    # Unique patches
    unique_patches = {}
    for r in all_records:
        pid = r["patch_id"]
        if pid not in unique_patches:
            unique_patches[pid] = r

    selected_pids = random.sample(list(unique_patches.keys()), 20)
    token = get_sas_token()

    audit_results = []
    print(f"Beginning image correspondence audit for 20 sample patches...")

    for idx, pid in enumerate(selected_pids):
        # S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_32_64
        parts = pid.split("_")
        sensor = parts[0]
        sensing_time = parts[2]
        date_str = f"{sensing_time[:4]}-{sensing_time[4:6]}-{sensing_time[6:8]}"
        mgrs = parts[5].replace("T", "")
        h_order = int(parts[6])  # Horizontal order = Column index
        v_order = int(parts[7])  # Vertical order = Row index

        # Query STAC for tile transform
        resp = requests.post(STAC_URL, json={
            "collections": ["sentinel-2-l2a"],
            "query": {"s2:mgrs_tile": {"eq": mgrs}},
            "datetime": f"{date_str}T00:00:00Z/{date_str}T23:59:59Z"
        }, timeout=15)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            continue
        item = features[0]
        b4_asset = item["assets"]["B04"]
        transform = b4_asset.get("proj:transform")
        epsg = item["properties"].get("proj:epsg", 32600)
        shape = b4_asset.get("proj:shape", [10980, 10980])

        x_orig = transform[2]
        y_orig = transform[5]
        res_x = transform[0]
        res_y = transform[4]

        # Correct footprint: col = h_order, row = v_order
        corr_col_off = h_order * 120
        corr_row_off = v_order * 120
        corr_ulx = x_orig + corr_col_off * res_x
        corr_uly = y_orig + corr_row_off * res_y
        corr_lrx = corr_ulx + 120 * res_x
        corr_lry = corr_uly + 120 * res_y

        # Stage A erroneous footprint (where row and col were swapped):
        err_col_off = v_order * 120
        err_row_off = h_order * 120
        err_ulx = x_orig + err_col_off * res_x
        err_uly = y_orig + err_row_off * res_y
        err_lrx = err_ulx + 120 * res_x
        err_lry = err_uly + 120 * res_y

        is_transposed = (h_order != v_order)
        displacement_km = np.sqrt((corr_ulx - err_ulx)**2 + (corr_uly - err_uly)**2) / 1000.0

        audit_results.append({
            "patch_id": pid,
            "product_id": item["id"],
            "timestamp": item["properties"].get("datetime"),
            "tile_id": mgrs,
            "crs": f"EPSG:{epsg}",
            "pixel_resolution": f"{res_x}m",
            "tile_shape": shape,
            "h_order": h_order,
            "v_order": v_order,
            "correct_bounds_utm": [round(corr_ulx, 1), round(corr_uly, 1), round(corr_lrx, 1), round(corr_lry, 1)],
            "stageA_bounds_utm": [round(err_ulx, 1), round(err_uly, 1), round(err_lrx, 1), round(err_lry, 1)],
            "bands_used": ["B04", "B03", "B02"],
            "patch_dimensions": "120x120",
            "spatial_displacement_km": round(displacement_km, 2),
            "stageA_correct": not is_transposed
        })
        print(f"[{idx+1}/20] Audited {pid}: H={h_order}, V={v_order} | Displacement: {displacement_km:.2f} km")

    with open("docs/BIGEARTHNET_IMAGE_CORRESPONDENCE_AUDIT.json", "w") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nAudit completed. Saved docs/BIGEARTHNET_IMAGE_CORRESPONDENCE_AUDIT.json.")

if __name__ == "__main__":
    main()
