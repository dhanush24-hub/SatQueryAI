# Accuracy Validation and Evaluation Report

**SatQuery AI — Prompt 6: Accuracy-First Remote-Sensing Intelligence**

---

## 1. Executive Summary

In Prompt 5.5, an empirical audit revealed that the active pretrained checkpoint (`HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff`) was degenerate on the benchmark distribution:
* Precision: 0.0212
* Recall: 0.0008
* F1: 0.0015
* IoU: 0.0008

To achieve reproducible, accurate, and scientifically defensible intelligence, we conducted rigorous benchmarking across **5 model configurations** on the standard LEVIR-CD benchmark dataset (`ericyu/LEVIRCD_Cropped256`).

---

## 2. Benchmark Results

| Model / Method | Parameters | Validation F1 | Validation IoU | Calibrated Threshold | Held-Out Test F1 | Held-Out Test IoU |
|---|---|---|---|---|---|---|
| **HZDR SiamUDiff (Pretrained)** | 1.35M | 0.0054 | 0.0028 | 0.50 | N/A (Degenerate) | N/A |
| **Analytical CVA Baseline** | 0 | 0.0453 | 0.0246 | Adaptive (2σ) | N/A | N/A |
| **Fine-Tuned FC-EF (Early Fusion)** | 1.94M | 0.2707 | 0.1843 | 0.40 | N/A | N/A |
| **Fine-Tuned SiamUnet_diff** | 1.35M | 0.2790 | 0.1912 | 0.20 | N/A | N/A |
| **AttentionChangeNet (Winner)** | **12.56M** | **0.4351** | **0.3426** | **0.40** | **0.2650** | **0.2142** |

---

## 3. Threshold Calibration Protocol

Thresholds were calibrated exclusively on the **held-out validation split** (128 pairs) across sweeps from 0.10 to 0.70.

For `AttentionChangeNet`:
* Threshold 0.20: F1 = 0.4227, IoU = 0.3245
* Threshold 0.30: F1 = 0.4322, IoU = 0.3374
* Threshold 0.35: F1 = 0.4345, IoU = 0.3410
* **Threshold 0.40 (Optimal): F1 = 0.4351, IoU = 0.3426**
* Threshold 0.50: F1 = 0.4314, IoU = 0.3390
* Threshold 0.60: F1 = 0.4152, IoU = 0.3221

---

## 4. Held-Out Test Split Performance

The winning model with optimal threshold (0.40) was evaluated **once** on 256 held-out test samples:
* **Precision**: 0.2484
* **Recall**: 0.3045
* **F1 Score**: 0.2650
* **IoU**: 0.2142

This represents an enormous improvement over the degenerate HZDR baseline (~176x increase in F1 score and ~267x increase in IoU).

---

## 5. Safety & Domain Safeguards

* **Domain Gate**: Evaluates modality (rejects SAR in optical change models), image dimensions (rejects <64x64), cloud/nodata cover (>60% returns `INSUFFICIENT_EVIDENCE`), and co-registration errors.
* **Fallback Guarantee**: If neural inference fails or local checkpoint is unavailable, the pipeline falls back gracefully to registered alternatives.
