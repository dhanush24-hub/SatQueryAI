#!/usr/bin/env python3
"""
Evaluates the generic zero-shot BLIP-VQA baseline on the EXACT Stage A
manually verified BigEarthNet.txt benchmark manifest (data/manifests/bigearthnet_txt_stageA_bench.json).
Computes overall accuracy, category breakdown (Presence, Area, Count, Adjacency),
format compliance, and confusion matrices.
Saves results to docs/BIGEARTHNET_STAGEA_BASELINE.json.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

BENCH_MANIFEST = "data/manifests/bigearthnet_txt_stageA_bench.json"
OUTPUT_FILE = "docs/BIGEARTHNET_STAGEA_BASELINE.json"
MODEL_ID = "Salesforce/blip-vqa-base"


def main():
    print(f"Loading Stage A benchmark manifest from {BENCH_MANIFEST}...")
    with open(BENCH_MANIFEST, "r") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} benchmark questions.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading generic baseline model: {MODEL_ID}...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_ID).to(device)
    model.eval()

    categories = ["presence", "area", "count", "adjacency"]
    cat_correct = {c: 0 for c in categories}
    cat_total = {c: 0 for c in categories}

    ans_balance = {"yes": 0, "no": 0}
    pred_balance = {"yes": 0, "no": 0, "other": 0}
    format_compliant = 0
    total_correct = 0
    evaluated = 0
    failures = 0
    latencies = []

    print("Beginning evaluation on authentic benchmark rasters...")
    t_start = time.time()

    for idx, rec in enumerate(records):
        raster_path = rec["local_raster_path"]
        if not os.path.exists(raster_path):
            failures += 1
            continue

        try:
            image = Image.open(raster_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {raster_path}: {e}")
            failures += 1
            continue

        q = rec["question"]
        gt = rec["answer"].lower().strip()
        task = rec.get("task", "unknown").lower()

        ans_balance[gt] = ans_balance.get(gt, 0) + 1

        t0 = time.time()
        inputs = processor(image, q, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        pred_raw = processor.decode(out[0], skip_special_tokens=True).strip().lower()
        latencies.append(time.time() - t0)

        # Standard binary extraction
        if "yes" in pred_raw and "no" not in pred_raw:
            pred = "yes"
        elif "no" in pred_raw and "yes" not in pred_raw:
            pred = "no"
        else:
            pred = pred_raw

        if pred in ["yes", "no"]:
            format_compliant += 1
            pred_balance[pred] += 1
        else:
            pred_balance["other"] += 1

        is_correct = (pred == gt)
        if is_correct:
            total_correct += 1

        evaluated += 1

        for c in categories:
            if c in task:
                cat_total[c] += 1
                if is_correct:
                    cat_correct[c] += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(records):
            print(f"  Processed {idx + 1}/{len(records)}: Acc so far = {total_correct/evaluated:.4f}")

    total_time = time.time() - t_start
    overall_acc = total_correct / evaluated if evaluated > 0 else 0.0

    cat_acc = {}
    for c in categories:
        cat_acc[c] = cat_correct[c] / cat_total[c] if cat_total[c] > 0 else 0.0

    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_ID,
        "adapter": None,
        "adaptation_status": "Generic Zero-Shot (Pre-Adaptation Baseline)",
        "split": "bench",
        "benchmark_sample_count": evaluated,
        "failures_or_unavailable": failures,
        "overall_binary_accuracy": round(overall_acc, 4),
        "category_accuracy": {
            "presence": round(cat_acc["presence"], 4),
            "area": round(cat_acc["area"], 4),
            "count": round(cat_acc["count"], 4),
            "adjacency": round(cat_acc["adjacency"], 4)
        },
        "category_counts": {c: {"total": cat_total[c], "correct": cat_correct[c]} for c in categories},
        "ground_truth_balance": ans_balance,
        "prediction_balance": pred_balance,
        "format_compliance_rate": round(format_compliant / evaluated, 4) if evaluated > 0 else 0.0,
        "mean_latency_ms": round(float(np.mean(latencies)) * 1000, 1) if latencies else 0.0,
        "total_evaluation_time_sec": round(total_time, 2)
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==================================================")
    print(f"STAGE A GENERIC BASELINE EVALUATION RESULTS")
    print(f"==================================================")
    print(f"Evaluated Samples:     {evaluated}")
    print(f"Overall Accuracy:      {overall_acc:.4f} ({total_correct}/{evaluated})")
    print(f"Presence Accuracy:     {cat_acc['presence']:.4f} ({cat_correct['presence']}/{cat_total['presence']})")
    print(f"Area Accuracy:         {cat_acc['area']:.4f} ({cat_correct['area']}/{cat_total['area']})")
    print(f"Counting Accuracy:     {cat_acc['count']:.4f} ({cat_correct['count']}/{cat_total['count']})")
    print(f"Adjacency Accuracy:    {cat_acc['adjacency']:.4f} ({cat_correct['adjacency']}/{cat_total['adjacency']})")
    print(f"Format Compliance:     {results['format_compliance_rate']:.4f}")
    print(f"Results saved to:      {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
