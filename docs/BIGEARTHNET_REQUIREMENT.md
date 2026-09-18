# SatQuery AI System Specification: BigEarthNet.txt Requirement Verification

**Project:** SatQuery AI (SatQuery AI Team)  
**Problem Statement:** ISRO / SatQuery AI — SatQuery AI Specification  
**Evaluation Scope:** Remote Sensing Vision-Language Adaptation & Multi-Sensor Benchmark Compliance

---

## 1. Exact Requirement from Source-of-Truth

The SatQuery AI (SatQuery AI) Problem Statement **SatQuery AI Specification: Intelligent Remote Sensing Analysis and Visual Question Answering** establishes the following requirements regarding data, models, and domain adaptation:

1. **BigEarthNet.txt / Benchmark Datasets:**
   - The system must support real Earth Observation (EO) benchmarks. Specifically, **BigEarthNet.txt** (the multi-sensor image-text extension of BigEarthNet v2.0 / reBEN curated by TU Berlin BIFOLD/RSiM) is the reference multi-sensor vision-language benchmark for satellite patch reasoning.
   - Datasets must maintain strict official split hygiene: `train`, `validation`, `test`, and `bench` (the curated held-out benchmark split).
2. **Domain Adaptation / Fine-Tuning:**
   - Generic commercial foundation models (e.g. standard BLIP, OWL-ViT, or generic web-pretrained VLMs) cannot be claimed as "remote-sensing adapted" without documented fine-tuning or parameter-efficient adaptation on authentic satellite data.
   - Merely providing a training script without executing training and generating a real checkpoint is explicitly non-compliant.
3. **Visual Question Answering (VQA):**
   - Must answer natural-language questions over satellite imagery regarding land-use/land-cover (LULC) presence, relative spatial extent (area), counting, and spatial adjacency.
4. **Visual Grounding / Referring Expression Detection:**
   - Must localize referenced terrestrial features with normalized bounding coordinates without hallucinating non-existent classes.
5. **Sentinel-1 (SAR) and Sentinel-2 (Optical/Multispectral):**
   - Sentinel-2: Ingests true-color (RGB: B04, B03, B02) and multispectral bands (B01-B12, 10m/20m/60m GSD).
   - Sentinel-1: Ingests dual-polarization SAR backscatter (VV and VH channels, Ground Range Detected / GRD).
   - SAR data must **never** be passed directly into optical RGB encoders as fake color imagery unless an explicit modality projection or SAR adapter is implemented.
6. **Evaluation Protocol:**
   - Strictly disjoint train and benchmark sets.
   - Zero test/benchmark threshold tuning.
   - Clear ablation reporting comparing generic baseline vs. RS-adapted models.

---

## 2. Current Implementation Status

| Capability Area | Current State in SatQuery AI | Verification Evidence |
|---|---|---|
| **Bi-Temporal Change Detection** | `AttentionChangeNet` (12.56M params) fully fine-tuned on LEVIR-CD | Held-out test $F_1 = 0.8459$, $\text{IoU} = 0.7330$ |
| **Physical SAR Preprocessing** | In-memory Enhanced Lee speckle filter ($5\times 5$, $\eta=1.0$), $\sigma^0_{\text{dB}}$ calibration | `sar_processing.py`, preserves raw files |
| **Optical + SAR Fusion** | `OpticalSarFusionAdapter` dual-specialist evidence reasoning | `optical_sar_fusion.py`, surfaces agreement & disagreement |
| **Cross-Modal Alignment Gate** | Normalized Mutual Information (NMI) + Sobel structural edge correlation | `cross_modal_alignment.py` (suppresses fusion on POOR) |
| **VQA & Grounding Models** | `Salesforce/blip-vqa-base` and `google/owlvit-base-patch32` | Operating zero-shot on RGB without RS adaptation |
| **RS Adaptation Status** | Marked **PARTIAL** in `docs/REQUIREMENT_MATRIX.md` | Training script exists (`train_rs_vlm_lora.py`), but no trained checkpoint |

---

## 3. Missing Components (To Be Implemented in Prompt 8.5)

1. **BigEarthNet.txt Ingestion & Manifests:**
   - Download/extract official `BigEarthNet.txt` annotations directly from `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`.
   - Create deterministic, leak-free manifests for `train`, `val`, and official `bench`.
2. **Quantitative Baseline Evaluation:**
   - Formally benchmark zero-shot BLIP-VQA and OWL-ViT on the official `bench` split to establish the empirical baseline.
3. **Real RS-VLM Adaptation (LoRA):**
   - Train a parameter-efficient adapter on a deterministic BigEarthNet.txt training split.
   - Perform model selection exclusively on `val`.
   - Save real model weights in `models/satquery_rs_vlm_v1/`.
4. **Official Benchmark Comparison:**
   - Evaluate adapted checkpoint on held-out `bench` split. Compare generic baseline vs. RS-adapted model across binary VQA categories (presence, area, counting, adjacency).
5. **System Integration & Registry:**
   - Register `satquery_rs_vlm_v1` in `model_registry.py` and hook into `controller.py` so natural-language single-scene queries route automatically to the adapted specialist.

---

## 4. Validation & Acceptance Protocol

The RS adaptation requirement will be promoted from **PARTIAL** to **PASS** only upon satisfaction of the following criteria:

- [x] Provenance of BigEarthNet.txt verified from official TU Berlin BIFOLD release (`arXiv:2603.29630`, `txt.bigearth.net`).
- [x] Reproducible dataset loader and manifests created (`data/manifests/bigearthnet_txt_stageA_*.json`).
- [x] No data leakage between `train`, `val`, and `bench` splits (100% disjoint, zero overlap).
- [x] Generic baseline quantitatively evaluated on `bench` (saved to `docs/BIGEARTHNET_STAGEA_BASELINE.json`, accuracy 0.5253).
- [x] Real LoRA fine-tuning executed on MPS, producing a physical checkpoint in `models/satquery_rs_vlm_v1/checkpoint/` (`adapter_model.safetensors`, 4.75 MB).
- [ ] Adapted model outperforms generic baseline on held-out benchmark (achieved +10.14% on Area and 100% format compliance, but overall accuracy 49.72% vs 52.53% due to conservative shift; baseline retained per Decision Gate).
- [x] Failure analysis and ablation documented (`docs/BIGEARTHNET_STAGEA_REPORT.md`).
- [x] Decision Gate executed: generic BLIP remains active in SatQuery agent pipeline to prevent regression.
- [x] All 92 backend tests pass (`92 passed in 5.42s`) and frontend builds cleanly (`npm run build` green).

**Current Status:** **PARTIAL (Phase A Completed & Honest Audit Retained)**

