import io
import json
import os
import sys
import numpy as np
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.models.siamunet_diff import load_siamunet_diff_model
from app.core.logging import logger


def evaluate_levircd_benchmark(
    num_samples: int = 50,
    threshold: float = 0.50,
    device: str = "cpu"
):
    """
    Evaluates SiamUnet_diff on the official LEVIR-CD256 validation set.
    Computes Precision, Recall, F1 Score, and IoU.
    """
    print(f"=== LEVIR-CD256 Benchmark Evaluation ===")
    print(f"Model: HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff (SiamUnet_diff)")
    print(f"Dataset: ericyu/LEVIRCD_Cropped256 (Validation Split)")
    print(f"Evaluating {num_samples} bi-temporal satellite pairs on {device} (threshold={threshold})...\n")

    # 1. Download/load dataset
    val_path = hf_hub_download(
        repo_id="ericyu/LEVIRCD_Cropped256",
        filename="data/val-00000-of-00001-d09d88a7419f2427.parquet",
        repo_type="dataset"
    )
    table = pq.read_table(val_path)
    total_available = len(table)
    num_samples = min(num_samples, total_available)

    # 2. Load model
    model = load_siamunet_diff_model(device=device)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    saved_sample = False

    for i in range(num_samples):
        row = table.slice(i, 1).to_pydict()
        imgA_bytes = row["imageA"][0]["bytes"]
        imgB_bytes = row["imageB"][0]["bytes"]
        lbl_bytes = row["label"][0]["bytes"]

        imgA = Image.open(io.BytesIO(imgA_bytes)).convert("RGB")
        imgB = Image.open(io.BytesIO(imgB_bytes)).convert("RGB")
        lbl = Image.open(io.BytesIO(lbl_bytes)).convert("L")

        arrA = np.array(imgA, dtype=np.float32) / 255.0  # (256, 256, 3)
        arrB = np.array(imgB, dtype=np.float32) / 255.0
        gt_mask = (np.array(lbl) > 128).astype(bool)     # True = change

        # Save first sample with significant change for real fixture verification
        if not saved_sample and np.sum(gt_mask) > 1000:
            os.makedirs("storage/fixtures", exist_ok=True)
            imgA.save("storage/fixtures/levir_t1_sample.png")
            imgB.save("storage/fixtures/levir_t2_sample.png")
            lbl.save("storage/fixtures/levir_gt_sample.png")
            print(f"Saved real benchmark pair fixture to storage/fixtures/levir_t1_sample.png and levir_t2_sample.png")
            saved_sample = True

        # Preprocess for model: (1, 3, 256, 256)
        tA = torch.from_numpy(arrA).permute(2, 0, 1).unsqueeze(0).to(device)
        tB = torch.from_numpy(arrB).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model(tA, tB)
            # Class 1 is change probability
            change_prob = torch.exp(log_probs[:, 1]).squeeze(0).cpu().numpy()
            pred_mask = (change_prob >= threshold)

        tp = np.sum(pred_mask & gt_mask)
        fp = np.sum(pred_mask & (~gt_mask))
        fn = np.sum((~pred_mask) & gt_mask)
        tn = np.sum((~pred_mask) & (~gt_mask))

        total_tp += int(tp)
        total_fp += int(fp)
        total_fn += int(fn)
        total_tn += int(tn)

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"Processed {i + 1}/{num_samples} pairs...")

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    results = {
        "benchmark_dataset": "LEVIR-CD256 (Validation)",
        "num_evaluated_pairs": num_samples,
        "threshold": threshold,
        "device": device,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "true_negatives": total_tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou": round(iou, 4)
    }

    print("\n=== Benchmark Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    os.makedirs("docs", exist_ok=True)
    with open("docs/TEMPORAL_BENCHMARK_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmark metrics to docs/TEMPORAL_BENCHMARK_RESULTS.json.")
    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_levircd_benchmark(num_samples=50, threshold=0.50, device=device)
