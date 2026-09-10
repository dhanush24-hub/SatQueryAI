#!/usr/bin/env python3
"""
Step 10: ONE-SHOT FINAL BENCHMARK EVALUATION for Stage 8.5B.
Strictly executed ONCE after the final model configuration is frozen.
Evaluates on the official 356 locked benchmark questions (data/manifests/bigearthnet_txt_stage8_5b_bench.json).

Computes:
- Overall binary VQA accuracy
- Balanced accuracy
- Precision, Recall, F1 for YES and NO
- Per-task accuracy (Presence, Area, Counting, Adjacency)
- Format compliance rate
- YES/NO prediction distribution
- Average latency per sample (ms)
- Hash of evaluated checkpoint
"""

import os
import sys
import json
import time
import hashlib
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel

BENCH_MANIFEST = "data/manifests/bigearthnet_txt_stage8_5b_bench.json"
BASE_MODEL_ID = "Salesforce/blip-vqa-base"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_REPORT = "docs/STAGE8_5B_FINAL_BENCHMARK_REPORT.json"


def compute_dir_sha256(dir_path: str) -> str:
    sha = hashlib.sha256()
    for root, _, files in sorted(os.walk(dir_path)):
        for f in sorted(files):
            fp = os.path.join(root, f)
            with open(fp, "rb") as fh:
                while chunk := fh.read(8192):
                    sha.update(chunk)
    return sha.hexdigest()


def evaluate_benchmark(adapter_dir: str):
    print(f"=== FINAL BENCHMARK EVALUATION (ONE-SHOT) ===", flush=True)
    print(f"Loading benchmark manifest: {BENCH_MANIFEST}", flush=True)
    with open(BENCH_MANIFEST, "r") as f:
        records = json.load(f)

    # Validate that all rasters exist
    for r in records:
        assert os.path.exists(r["local_raster_path"]), f"Missing raster: {r['local_raster_path']}"

    print(f"Loaded {len(records)} human-verified benchmark questions across {len(set(r['patch_id'] for r in records))} unique patches.", flush=True)

    ckpt_hash = compute_dir_sha256(adapter_dir) if os.path.isdir(adapter_dir) else "N/A"
    print(f"Adapter Directory: {adapter_dir}")
    print(f"Adapter SHA256: {ckpt_hash}")

    print(f"Loading {BASE_MODEL_ID} + adapter on {DEVICE}...", flush=True)
    processor = BlipProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(DEVICE)
    model.eval()

    total = len(records)
    correct = 0
    y_true = []
    y_pred = []
    latencies = []
    compliant_count = 0

    task_stats = {
        "presence": {"correct": 0, "total": 0},
        "area": {"correct": 0, "total": 0},
        "count": {"correct": 0, "total": 0},
        "adjacency": {"correct": 0, "total": 0}
    }

    t_start = time.time()
    with torch.no_grad():
        for idx, rec in enumerate(records):
            img_path = rec["local_raster_path"]
            img = Image.open(img_path).convert("RGB")
            q = rec["question"]
            gt = rec["answer"].strip().lower()
            task = rec["task"]

            inputs = processor(images=img, text=q, return_tensors="pt").to(DEVICE)

            t0 = time.time()
            out = model.generate(
                **inputs,
                max_new_tokens=5,
                num_beams=1,
                do_sample=False
            )
            lat = (time.time() - t0) * 1000.0
            latencies.append(lat)

            pred_raw = processor.decode(out[0], skip_special_tokens=True).strip().lower()

            # Binary format compliance: does it cleanly yield 'yes' or 'no'?
            if pred_raw in ["yes", "no"]:
                compliant_count += 1
                pred_bin = pred_raw
            elif "yes" in pred_raw and "no" not in pred_raw:
                pred_bin = "yes"
            elif "no" in pred_raw and "yes" not in pred_raw:
                pred_bin = "no"
            else:
                pred_bin = pred_raw

            y_true.append(gt)
            y_pred.append(pred_bin)

            is_correct = (pred_bin == gt)
            if is_correct:
                correct += 1

            if task in task_stats:
                task_stats[task]["total"] += 1
                if is_correct:
                    task_stats[task]["correct"] += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"  Evaluated {idx + 1}/{total} samples (current acc: {correct / (idx + 1):.4f})...", flush=True)

    total_eval_time = time.time() - t_start
    overall_acc = round(correct / total, 4)

    # Yes/No metrics
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "yes" and yp == "yes")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "no" and yp == "yes")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "yes" and yp == "no")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "no" and yp == "no")

    prec_yes = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    rec_yes = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1_yes = round(2 * prec_yes * rec_yes / (prec_yes + rec_yes), 4) if (prec_yes + rec_yes) > 0 else 0.0

    prec_no = round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0.0
    rec_no = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    f1_no = round(2 * prec_no * rec_no / (prec_no + rec_no), 4) if (prec_no + rec_no) > 0 else 0.0

    balanced_acc = round((rec_yes + rec_no) / 2.0, 4)

    task_accuracies = {
        k: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0
        for k, v in task_stats.items()
    }

    pred_dist = {
        "yes": sum(1 for p in y_pred if p == "yes"),
        "no": sum(1 for p in y_pred if p == "no"),
        "other": sum(1 for p in y_pred if p not in ["yes", "no"])
    }

    report = {
        "benchmark_file": BENCH_MANIFEST,
        "total_samples": total,
        "adapter_dir": adapter_dir,
        "adapter_sha256": ckpt_hash,
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "format_compliance_rate": round(compliant_count / total, 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "total_evaluation_time_sec": round(total_eval_time, 2),
        "metrics_yes": {
            "precision": prec_yes,
            "recall": rec_yes,
            "f1": f1_yes
        },
        "metrics_no": {
            "precision": prec_no,
            "recall": rec_no,
            "f1": f1_no
        },
        "per_task_accuracy": task_accuracies,
        "prediction_distribution": pred_dist,
        "comparison": {
            "generic_blip_baseline": 0.5253,
            "stage_a_adapted": 0.4972,
            "stage_8_5b_adapted": overall_acc,
            "delta_over_generic": round(overall_acc - 0.5253, 4),
            "delta_over_stage_a": round(overall_acc - 0.4972, 4)
        }
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nBenchmark Evaluation Finished! Report saved to {OUTPUT_REPORT}")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}% (Generic BLIP: 52.53%, Stage A: 49.72%)")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"Predictions: {pred_dist}")
    print(f"Per-Task Accuracies: {task_accuracies}")
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1:
        adapter_path = sys.argv[1]
    else:
        adapter_path = "models/stage8_5b_recovery/experiment_c_balanced_sampler_lr3e-5"
    evaluate_benchmark(adapter_path)
