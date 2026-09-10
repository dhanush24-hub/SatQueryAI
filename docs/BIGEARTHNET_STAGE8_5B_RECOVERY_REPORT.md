# BigEarthNet.txt Stage 8.5B: S2-RGB Accuracy Recovery Report

## Executive Summary
This document provides the complete, scientifically verified results for **Stage 8.5B — S2-RGB Accuracy Recovery** on the authentic BigEarthNet.txt benchmark (arXiv:2603.29630).

### Key Outcomes
- **Image Transposition Bug Proven and Rectified**: Mathematical and spatial audit proved that Stage A had swapped row and column indices (`col_off = V*120`, `row_off = H*120`), displacing patches 10.3 km to 147.2 km across the tile diagonal. All imagery was re-extracted with verified coordinates `(col_off = H*120, row_off = V*120)`.
- **Dataset Expanded 5x**: Scaled from 200 train patches to 1,000 unique patches (7,644 questions) across Finland and Portugal, and 300 validation patches (2,396 questions) across Serbia and Finland.
- **Catastrophic Answer Collapse Eliminated**:
  - Stage A: 67 Yes / 289 No (extreme negative collapse)
  - Generic BLIP: 288 Yes / 48 No (extreme positive bias)
  - Stage 8.5B: **177 Yes / 179 No** (near-perfect 50/50 balance)
- **Supervision & Memorization Proven**: 16-sample tiny overfit test drove loss from 0.4944 to 0.0201, reaching 93.8% accuracy.
- **Benchmark Performance**: Evaluated **strictly once** on the locked 356 benchmark questions:
  - Generic BLIP: 52.53%
  - Stage A Adapted: 49.72%
  - Stage 8.5B Adapted: **50.28%** (Balanced Accuracy: 50.32%)
- **Decision Gate Outcome**: Because overall accuracy (50.28%) is $\le$ generic BLIP (52.53%) and below the >60% target, adaptation status remains **PARTIAL**, generic BLIP is retained in production, and S1+S2 is deferred.

---

## 1. Image Correspondence Audit
- **Audit Script**: `scripts/audit_image_correspondence.py`
- **Audit Data**: `docs/BIGEARTHNET_IMAGE_CORRESPONDENCE_AUDIT.json`
- **Findings**:
  - BigEarthNet patch naming: `<S2_NAME>_<H-Order>_<V-Order>`.
  - H-Order is the horizontal/column index ($X$ offset = $H \times 120$ px).
  - V-Order is the vertical/row index ($Y$ offset = $V \times 120$ px).
  - In Stage A, `row = int(parts[6])` and `col = int(parts[7])` inverted these axes, causing window displacement of $\sqrt{(H - V)^2 \times 2} \times 1200\text{ m}$.
  - Verified across 20 randomly sampled patches: 19 of 20 had $H \neq V$ with displacements between 10.3 km and 147.2 km.
- **Remediation**:
  - Fixed formula: `col_offset = h_order * 120`, `row_offset = v_order * 120`.
  - Deleted all old transposed PNGs.
  - Re-acquired 1,332 authentic patches directly from Microsoft Planetary Computer Sentinel-2 L2A COGs with SAS tokens.

---

## 2. Dataset Expansion & Manifests
- **Pipeline**: `scripts/acquire_stage8_5b_dataset.py`
- **Total Storage**: 31.16 MB across 1,332 `.png` files (120x120 pixels, 3 bands B04, B03, B02).
- **Subsets**:
  - **Train**: 1,000 unique patches, 7,644 questions (500 from T35VNK_20180525, 500 from T29SNC_20180515).
  - **Validation**: 300 unique patches, 2,396 questions (150 from T34TEP_20180502, 150 from T35VNL_20180525).
  - **Benchmark**: 32 unique patches, 356 questions (T29SND_20180515) — **LOCKED** until final one-shot evaluation.
- **Disjointness**: 100% zero leakage verified across all splits.

---

## 3. Train/Val Distributions
- **Train (7,644 questions)**:
  - Yes: 3,644 (47.67%), No: 4,000 (52.33%)
  - Tasks: Area (26.16%), Count (26.16%), Presence (26.16%), Adjacency (21.51%)
  - Top Land Cover: Mixed forest (483), Coniferous forest (454), Marine waters (373), Broad-leaved forest (285).
- **Validation (2,396 questions)**:
  - Yes: 1,196 (49.92%), No: 1,200 (50.08%)
  - Tasks: Area (25.04%), Count (25.04%), Presence (25.04%), Adjacency (24.87%)

---

## 4. Stage A Bias Root Cause Analysis
- **Problem**: Stage A predicted 67 Yes vs 289 No (severe negative bias).
- **Causes**:
  1. **Image Transposition**: The ground truth labels answered questions about the true patch $(H, V)$, but the model was shown $(V, H)$. Because land cover classes are sparse across a 110 km tile, the queried feature was almost never present in the shifted patch.
  2. **Gradient Skew**: The model received heavy negative gradients whenever it attempted to predict positive classes, systematically depressing weights for "yes" tokens.
  3. **Padding Bug**: `pad_sequence` padded with 0 without `ignore_index=-100`, penalizing the loss across non-target positions.

---

## 5. Tokenization & Supervision Audit
- **BLIP Tokenizer**: BertTokenizer
- Target IDs:
  - `yes`: `[101, 2748, 102]` ([CLS] yes [SEP])
  - `no`:  `[101, 2053, 102]` ([CLS] no [SEP])
- Masking: `labels[labels == pad_token_id] = -100` so cross-entropy loss applies strictly to answer tokens.

---

## 6. Tiny Overfit Test (16 Samples)
- Script: `scripts/test_tiny_overfit.py`
- Samples: 8 Yes, 8 No from authentic train patches.
- Initial accuracy: 10/16 (62.5%)
- Optimization: 30 steps with AdamW lr=5e-4 on Apple Silicon `mps`.
- Loss trajectory: 0.4944 $\to$ 0.3272 $\to$ 0.2530 $\to$ 0.0596 $\to$ 0.0402 $\to$ 0.0201.
- Final accuracy: **15/16 (93.8%)** with predictions `['yes'*8, 'no'*7, 'yes']`. Confirms memorization capability and valid supervision pipeline.

---

## 7. Image Preprocessing Comparison (Validation Only)
- Script: `scripts/test_preprocessing_comparison.py`
- Evaluated on 100 validation samples:
  - **Strategy A (Standard Reflectance 0-3000 linear clip)**: **54.00%**
  - **Strategy B (2-98% Percentile Stretch)**: **54.00%**
  - **Strategy C (Gamma 1.1 Shadow Stretch)**: **52.00%**
- Strategy A chosen: Preserves physical radiometric linearity across bands.

---

## 8. Validation Ablation Experiments (Validation Only)
- Script: `scripts/train_stage8_5b_recovery.py`
- Comparison table:

| Configuration | Val Acc | Balanced Acc | F1 Yes | F1 No | Pred Dist (Yes/No) | Val Loss |
|---|---|---|---|---|---|---|
| Pre-training (Generic BLIP) | 53.75% | 53.88% | 0.6842 | 0.1819 | 222 / 15 | 0.6187 |
| Exp A (LoRA LR=1e-4, Std) | 52.08% | 52.09% | 0.3784 | 0.6102 | 65 / 175 | 0.4579 |
| Exp B (LoRA LR=3e-5, Std) | 55.42% | 55.42% | 0.5486 | 0.5597 | 117 / 123 | 0.4193 |
| **Exp C (LoRA LR=3e-5, Balanced)** | **57.08%** | **57.08%** | **0.6171** | **0.5118** | **149 / 91** | **0.4218** |

- **Winner**: **Experiment C** achieved the highest overall validation accuracy (57.08%), balanced accuracy (57.08%), and per-task robustness (Presence 63.33%, Count 58.33%, Area 58.33%).

---

## 9. Final Frozen Configuration
- **Model**: Salesforce/blip-vqa-base with PEFT LoRA
- **LoRA Hyperparameters**: $r=8, \alpha=16, \text{dropout}=0.05, \text{targets}=[\text{"query"}, \text{"value"}]$
- **Learning Rate**: $3 \times 10^{-5}$
- **Sampler**: WeightedRandomSampler balancing across (Task, Answer)
- **Preprocessing**: Surface reflectance $0-3000 \to [0, 255]$ uint8
- **Decoding**: Greedy (`max_new_tokens=5, num_beams=1, do_sample=False`)
- **Frozen Checkpoint Path**: `models/satquery_rs_vlm_v1/final_checkpoint`
- **SHA256**: `2c98ad3d3e63347b991d6e83bc21a17655e2f1202e3cacebdf1195733a578120`

---

## 10. One-Shot Final Benchmark Evaluation
- Script: `scripts/evaluate_stage8_5b_final.py`
- Manifest: `data/manifests/bigearthnet_txt_stage8_5b_bench.json` (356 human-verified questions)
- Executed **strictly once**:
  - **Overall Accuracy**: **50.28%** (179/356)
  - **Balanced Accuracy**: **50.32%**
  - **Format Compliance**: **100%** (356/356)
  - **Latency**: Mean 270.56 ms, Median 261.27 ms, p95 366.12 ms
  - **Predictions**: **177 Yes / 179 No / 0 Other**
  - **Per-Task Accuracy**:
    - Presence: 31.37% (16/51)
    - Area: **65.22%** (60/92)
    - Counting: 47.46% (56/118)
    - Adjacency: 50.85% (47/95)

---

## 11. Comparison: Generic vs Stage A vs Recovery

| Metric | Generic BLIP | Stage A Adapted | Stage 8.5B Recovery |
|---|---|---|---|
| **Overall Accuracy** | **52.53%** | 49.72% | 50.28% |
| **Balanced Accuracy** | 50.00% (approx) | 49.65% | **50.32%** |
| **Prediction Balance** | 288 Yes / 48 No | 67 Yes / 289 No | **177 Yes / 179 No** |
| **Area Task Acc** | 53.26% | 48.91% | **65.22%** |
| **Compliance Rate** | 94.38% | 100.0% | **100.0%** |
| **Catastrophic Collapse** | Severe positive bias | Severe negative bias | **None (balanced)** |

---

## 12. Decision Gate & Status
- **Decision Gate Criteria**: PASS only if $>52.53\%$ and target $>60\%$.
- **Result**: 50.28% is $\le 52.53\%$.
- **Status**: **PARTIAL**.
- **Action**: Retain Generic BLIP as the production VLM default. Do not deploy adapted checkpoint to production.
- **Ready for S1+S2 Stage B**: **NO** (stopped per instruction).
