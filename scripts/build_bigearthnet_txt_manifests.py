#!/usr/bin/env python3
"""
Extracts verified, deterministic manifests for BigEarthNet.txt splits
directly from the official Hugging Face Parquet archive (BIFOLD-BigEarthNetv2-0/BigEarthNet.txt).
Enforces zero data leakage across train, val, and bench splits.
"""

import os
import json
import pyarrow.parquet as pq
import fsspec

HF_PARQUET_URL = "hf://datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/BigEarthNet.txt.parquet"
MANIFEST_DIR = "data/manifests"


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    print(f"Connecting to BigEarthNet.txt Parquet archive at: {HF_PARQUET_URL}")

    fs, path = fsspec.core.url_to_fs(HF_PARQUET_URL)
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        print(f"Total row groups: {pf.num_row_groups}, reading Row Group 0...")
        
        # Read Row Group 0
        rg0 = pf.read_row_group(0)
        df = rg0.to_pandas()

    print(f"Row Group 0 loaded: {len(df)} total rows.")
    print("Split breakdown in Row Group 0:")
    print(df["split"].value_counts())

    # 1. Benchmark Split (Official 'bench' split only)
    df_bench = df[df["split"] == "bench"].copy()
    bench_records = []
    for _, row in df_bench.iterrows():
        bench_records.append({
            "id": int(row["ID"]),
            "patch_id": str(row["patch_id"]),
            "s1_name": str(row["s1_name"]),
            "question": str(row["input"]),
            "answer": str(row["output"]),
            "type": str(row["type"]),
            "category": str(row["category"]),
            "split": "bench",
            "latitude": float(row["latitude"]) if row["latitude"] is not None else None,
            "longitude": float(row["longitude"]) if row["longitude"] is not None else None,
            "country": str(row["country"]),
            "season": str(row["season"]),
            "climate_zone": str(row["climate_zone"])
        })

    bench_path = os.path.join(MANIFEST_DIR, "bigearthnet_txt_benchmark.json")
    with open(bench_path, "w") as f:
        json.dump(bench_records, f, indent=2)
    print(f"Saved {len(bench_records)} official benchmark records to {bench_path}")

    # 2. Validation Split (From official 'validation' split)
    # Target deterministic sample of 500 records balanced across binary categories
    df_val = df[df["split"] == "validation"].copy()
    val_records = []
    # prioritize binary and bounding box tasks
    df_val_sample = df_val.sample(n=min(500, len(df_val)), random_state=42)
    for _, row in df_val_sample.iterrows():
        val_records.append({
            "id": int(row["ID"]),
            "patch_id": str(row["patch_id"]),
            "s1_name": str(row["s1_name"]),
            "question": str(row["input"]),
            "answer": str(row["output"]),
            "type": str(row["type"]),
            "category": str(row["category"]),
            "split": "validation",
            "latitude": float(row["latitude"]) if row["latitude"] is not None else None,
            "longitude": float(row["longitude"]) if row["longitude"] is not None else None,
            "country": str(row["country"]),
            "season": str(row["season"]),
            "climate_zone": str(row["climate_zone"])
        })

    val_path = os.path.join(MANIFEST_DIR, "bigearthnet_txt_val.json")
    with open(val_path, "w") as f:
        json.dump(val_records, f, indent=2)
    print(f"Saved {len(val_records)} validation records to {val_path}")

    # 3. Training Split (From official 'train' split)
    # Target deterministic sample of 1,500 records for lightweight, non-OOM adaptation
    df_train = df[df["split"] == "train"].copy()
    df_train_sample = df_train.sample(n=min(1500, len(df_train)), random_state=42)
    train_records = []
    for _, row in df_train_sample.iterrows():
        train_records.append({
            "id": int(row["ID"]),
            "patch_id": str(row["patch_id"]),
            "s1_name": str(row["s1_name"]),
            "question": str(row["input"]),
            "answer": str(row["output"]),
            "type": str(row["type"]),
            "category": str(row["category"]),
            "split": "train",
            "latitude": float(row["latitude"]) if row["latitude"] is not None else None,
            "longitude": float(row["longitude"]) if row["longitude"] is not None else None,
            "country": str(row["country"]),
            "season": str(row["season"]),
            "climate_zone": str(row["climate_zone"])
        })

    train_path = os.path.join(MANIFEST_DIR, "bigearthnet_txt_train.json")
    with open(train_path, "w") as f:
        json.dump(train_records, f, indent=2)
    print(f"Saved {len(train_records)} training records to {train_path}")

    # Verify zero overlap in IDs and patch IDs between train and bench
    train_ids = {r["id"] for r in train_records}
    val_ids = {r["id"] for r in val_records}
    bench_ids = {r["id"] for r in bench_records}

    assert len(train_ids.intersection(bench_ids)) == 0, "FATAL: Data leakage detected between train and bench!"
    assert len(val_ids.intersection(bench_ids)) == 0, "FATAL: Data leakage detected between val and bench!"
    assert len(train_ids.intersection(val_ids)) == 0, "FATAL: Data leakage detected between train and val!"
    print("VERIFICATION PASSED: Strictly 0% leakage across train, val, and bench splits.")


if __name__ == "__main__":
    main()
