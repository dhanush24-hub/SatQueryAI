#!/usr/bin/env python3
"""
Step 8: Image Preprocessing Comparison on VALIDATION ONLY.
Compares 3 Sentinel-2 surface reflectance preprocessing methods:
  A. Current linear clip: rgb / 3000.0 clipped to [0, 1]
  B. Percentile stretch: 2nd to 98th percentile contrast enhancement
  C. Dynamic range reflectance: rgb / 4000.0 with gamma 1.1 correction

Evaluates on 100 VALIDATION samples using base BLIP-VQA to determine
which preprocessing provides the highest zero-shot / feature fidelity.
STRICT RULE: Validation only. Benchmark is never accessed.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

MODEL_ID = "Salesforce/blip-vqa-base"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
VAL_MANIFEST = "data/manifests/bigearthnet_txt_stage8_5b_val.json"


def preprocess_strategy_a(raw_png_path):
    # Standard baseline: 0-3000 reflectance as stored in PNG
    return Image.open(raw_png_path).convert("RGB")


def preprocess_strategy_b(raw_png_path):
    # Percentile 2-98 stretch
    im = Image.open(raw_png_path).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 > p2:
        arr_stretched = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_stretched)
    return im


def preprocess_strategy_c(raw_png_path):
    # Gamma 1.1 correction for shadow enhancement
    im = Image.open(raw_png_path).convert("RGB")
    arr = np.array(im, dtype=np.float32) / 255.0
    arr_gamma = np.power(arr, 1.0 / 1.1)
    arr_uint8 = np.clip(arr_gamma * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_uint8)


def evaluate_strategy(model, processor, records, preprocess_fn, name):
    correct = 0
    total = 0
    yes_count = 0
    no_count = 0

    with torch.no_grad():
        for rec in records:
            img = preprocess_fn(rec["local_raster_path"])
            q = rec["question"]
            gt = rec["answer"].strip().lower()

            inputs = processor(images=img, text=q, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=5)
            pred = processor.decode(out[0], skip_special_tokens=True).strip().lower()

            pred_bin = "yes" if "yes" in pred and "no" not in pred else ("no" if "no" in pred and "yes" not in pred else pred)
            if pred_bin == "yes":
                yes_count += 1
            elif pred_bin == "no":
                no_count += 1

            if pred_bin == gt:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    return {
        "strategy": name,
        "total_evaluated": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "yes_count": yes_count,
        "no_count": no_count,
        "yes_ratio": round(yes_count / total, 4) if total > 0 else 0.0
    }


def main():
    if not os.path.exists(VAL_MANIFEST):
        print(f"Waiting for {VAL_MANIFEST} to be generated...", flush=True)
        return

    with open(VAL_MANIFEST, "r") as f:
        records = json.load(f)

    # Filter to existing images
    valid_records = [r for r in records if os.path.exists(r["local_raster_path"])][:100]
    if len(valid_records) < 50:
        print(f"Only {len(valid_records)} valid records available. Waiting for rasters...", flush=True)
        return

    print(f"Loaded {len(valid_records)} validation samples for preprocessing comparison.")
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    res_a = evaluate_strategy(model, processor, valid_records, preprocess_strategy_a, "A (Standard Reflectance 0-3000)")
    print("Result A:", res_a)

    res_b = evaluate_strategy(model, processor, valid_records, preprocess_strategy_b, "B (2-98% Percentile Stretch)")
    print("Result B:", res_b)

    res_c = evaluate_strategy(model, processor, valid_records, preprocess_strategy_c, "C (Gamma 1.1 Shadow Stretch)")
    print("Result C:", res_c)

    results = {
        "dataset": "BigEarthNet.txt Validation",
        "samples": len(valid_records),
        "strategies": [res_a, res_b, res_c]
    }
    out_file = "docs/STAGE8_5B_PREPROCESSING_COMPARISON.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
