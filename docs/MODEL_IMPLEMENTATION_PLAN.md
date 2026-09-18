# Model Implementation Plan: Single-Image Remote-Sensing VQA & Grounding

**Project:** SatQuery AI — SatQuery AI Team (SatQuery AI System Architecture)  
**Phase:** Prompt 3 of 8 (Single-Image VQA + Grounding Vertical Slice)  
**Date:** September 2026

---

## 1. Compute & Model Environment Audit

An audit of the local development and execution environment was conducted:

| Parameter | Observed Value | Operational Implication |
| :--- | :--- | :--- |
| **Operating System** | macOS 26.6.2 (Mach-O) | POSIX-compliant; UNIX sockets & signals supported |
| **CPU Architecture** | arm64 (Apple Silicon) | Native NEON vector acceleration |
| **Python Version** | 3.12.13 (`.venv`), 3.13.7 (system) | Full compatibility with PyTorch 2.x and Transformers 4.x |
| **Total System RAM** | 8.0 GB | Constrained unified memory |
| **Available Free RAM** | ~1.16 GB – 1.8 GB free | Cannot load 7B/13B parameter models in unquantized fp16 (>14 GB required) |
| **GPU / Accelerator** | Apple Metal (MPS) available; CUDA unavailable | MPS acceleration enabled for PyTorch tensors; CUDA kernels unavailable |
| **Available Disk Space** | ~100.0 GB free | Ample room for model cache (<5 GB) and test datasets |
| **Hugging Face Access** | Anonymous public access available; token required for gated checkpoints | Non-gated open models must be used for default local execution |

### Honest Assessment of Hardware Constraints
A 7B model such as `MBZUAI/geochat-7b` requires ~14 GB of memory in fp16, or ~5.5 GB in 4-bit quantization with `bitsandbytes` (which requires CUDA compilation). Attempting to load a 7B model locally on an 8 GB unified memory Mac with ~1.2 GB free will trigger OOM kernel kills.

Therefore, our architecture implements:
1. **Primary Local Inference Stack**:
   - **VQA**: `Salesforce/blip-vqa-base` (~850 MB weights). Loads in seconds, operates within available RAM, and runs fast on CPU or MPS.
   - **Text-Guided Grounding**: `google/owlvit-base-patch32` (~600 MB weights). Open-vocabulary text-conditioned zero-shot grounding that accepts natural language queries ("water body", "runway", "building", "storage tank") and outputs real bounding boxes with model scores.
2. **Modular GPU Deployment Path**:
   - `GeoChatAdapter` interface registered in `ModelToolRegistry` for deployment on CUDA-enabled workstations/clusters.
3. **Reproducible Adaptation Pipeline**:
   - BigEarthNet.txt adaptation script (`train_adaptation.py`) using PEFT/LoRA. Status is clearly documented as `NOT YET COMPLETED` without claiming false fine-tuning until executed on GPU hardware.

---

## 2. Model Evaluation & Selection

We evaluated candidate open models for remote sensing:

### Candidate 1: `MBZUAI/geochat-7b`
- **Source**: Hugging Face (`MBZUAI/geochat-7b`), CVPR 2024
- **License**: Apache-2.0
- **Strengths**: Specifically fine-tuned on 100k remote sensing dialogues, VQA, and referring expression grounding.
- **Limitations**: 13 GB weight footprint. Requires 14–16 GB VRAM / RAM. Blocked on 8 GB local machine.
- **Role**: Integrated as an optional GPU adapter in `ModelToolRegistry`.

### Candidate 2: `Salesforce/blip-vqa-base` (Selected for Local VQA)
- **Source**: Hugging Face (`Salesforce/blip-vqa-base`), ICML 2022
- **License**: BSD-3-Clause
- **Checkpoint**: `Salesforce/blip-vqa-base` (~850 MB)
- **Strengths**: Native Visual Question Answering architecture combining ViT encoder and auto-regressive multimodal text decoder. Runs smoothly on CPU/MPS with <1.5 GB memory overhead.
- **Input**: Image (RGB tensor normalized `(3, H, W)`) + natural language question string.
- **Output**: Generative text answer string.
- **Limitations**: Trained on general domain VQA; remote-sensing adaptation via LoRA is required for domain-specific terminology.

### Candidate 3: `google/owlvit-base-patch32` (Selected for Text-Guided Grounding)
- **Source**: Hugging Face (`google/owlvit-base-patch32`), ECCV 2022
- **License**: Apache-2.0
- **Checkpoint**: `google/owlvit-base-patch32` (~600 MB)
- **Strengths**: Genuine open-vocabulary text-conditioned object detection and visual grounding. Accepts natural language text queries without fixed label sets and outputs real spatial bounding boxes and confidence scores.
- **Input**: Image (RGB tensor) + list of target text queries.
- **Output**: Bounding boxes `[x1, y1, x2, y2]` in normalized coordinates `[0, 1]` with classification logits and sigmoid scores.
- **Limitations**: Requires coordinate transformation and bounding box clamping when mapping back to original raster pixel dimensions and WGS84 geographic footprints.

---

## 3. Adaptation Strategy (BigEarthNet)

The official Problem Statement requires at least one adapted visual/VLM component using BigEarthNet.txt or another open remote-sensing dataset.

### Dataset Details
- **Dataset**: BigEarthNet-S2 (Sentinel-2 benchmark with 43-class Corine Land Cover nomenclature) / BigEarthNet.txt split files.
- **License**: Community Data License Agreement – Permissive (CDLA-Permissive-1.0).
- **Strategy**: Low-Rank Adaptation (PEFT / LoRA) applied to the vision encoder projection and cross-attention text projection layers (`q_proj`, `v_proj`).
- **Pipeline Implementation**:
  - `backend/app/services/models/adaptation/train_adaptation.py`
  - `backend/app/services/models/adaptation/evaluate_adaptation.py`
- **Current Status**: **NOT YET COMPLETED (Runnable Pipeline Delivered)**.
  The adaptation training script is fully implemented and reproducible. Execution of multi-epoch training is deferred until GPU compute is attached, adhering strictly to the non-negotiable rule against false fine-tuning claims.

---

## 4. Coordinate Transformation Pipeline

To bridge raw model outputs with geospatial GIS layers:
1. **Model Coordinates**: Normalized `[0, 1]` bounding boxes `[ymin, xmin, ymax, xmax]` or `[x1, y1, x2, y2]`.
2. **Original Raster Pixel Coordinates**: Scaled by raster width and height:
   $$X_{pixel} = x \times W_{raster}, \quad Y_{pixel} = y \times H_{raster}$$
3. **Canvas Normalized Percentages**: `[x_pct, y_pct, w_pct, h_pct]` where coordinates range `0.0 – 100.0%` for direct CSS overlay rendering on `ImageryCanvas`.
4. **Geographic Bounding Box (WGS84)**: Derived via the raster's affine geotransform matrix ($A$):
   $$\begin{pmatrix} X_{geo} \\ Y_{geo} \end{pmatrix} = \begin{pmatrix} a & b & c \\ d & e & f \end{pmatrix} \begin{pmatrix} X_{pixel} \\ Y_{pixel} \\ 1 \end{pmatrix}$$
   When the native CRS is projected (e.g. UTM), coordinates are transformed to WGS84 (`EPSG:4326`) via PyProj/Rasterio.
