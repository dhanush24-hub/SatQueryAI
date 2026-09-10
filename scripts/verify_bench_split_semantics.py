#!/usr/bin/env python3
"""
Verifies the exact relationship between BigEarthNet.txt `bench` split
and the original BigEarthNet v2.0 (reBEN) `test` partition.
"""

import pandas as pd
import pyarrow.parquet as pq

def main():
    print("Loading BigEarthNet v2.0 full metadata (metadata_v2.parquet)...")
    v2_meta = pd.read_parquet("data/bigearthnet/metadata_v2.parquet")
    print(f"V2 Metadata loaded: {len(v2_meta)} patches.")
    v2_split_map = dict(zip(v2_meta["patch_id"], v2_meta["split"]))

    print("Opening BigEarthNet.txt.parquet...")
    pf = pq.ParquetFile("data/bigearthnet/BigEarthNet.txt.parquet")
    print(f"Total row groups: {pf.num_row_groups}, total rows: {pf.metadata.num_rows}")

    # Read essential columns
    df_txt = pf.read(columns=["ID", "patch_id", "s1_name", "split", "type", "category", "country"]).to_pandas()
    print(f"Loaded BigEarthNet.txt: {len(df_txt)} rows.")
    print("BigEarthNet.txt split distribution:")
    print(df_txt["split"].value_counts())

    bench_df = df_txt[df_txt["split"] == "bench"]
    bench_patches = bench_df["patch_id"].unique()
    print(f"\nBenchmark annotations: {len(bench_df)}")
    print(f"Benchmark unique patch IDs: {len(bench_patches)}")

    # Check against V2 split map
    v2_splits_of_bench = [v2_split_map.get(pid, "NOT_FOUND") for pid in bench_patches]
    v2_counts = pd.Series(v2_splits_of_bench).value_counts()
    print("\nOriginal BigEarthNet v2.0 split of all benchmark patches:")
    print(v2_counts)

    # Verify 100% test origin
    pct_test = (v2_counts.get("test", 0) / len(bench_patches)) * 100
    print(f"\nPercentage of benchmark patches originating from BigEarthNet v2.0 test split: {pct_test:.2f}%")
    assert pct_test == 100.0, "FATAL: Benchmark contains patches from outside the test split!"
    print("PROOF CONFIRMED: 100% of benchmark patches were selected exclusively from the BigEarthNet v2.0 test split!")

if __name__ == "__main__":
    main()
