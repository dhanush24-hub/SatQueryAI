# Model Provenance & Registry Specification

**Project:** SatQuery AI — SatQuery AI Team (SatQuery AI / SatQuery AI Specification)  
**Document Version:** 1.0.0  
**Phase:** Prompt 3 (Single-Image VQA & Grounding)

---

## 1. Registered Models & Checkpoints

### 1.1 Single-Image Visual Question Answering (`SINGLE_VQA`)

```yaml
model_id: "Salesforce/blip-vqa-base"
framework: "PyTorch / Hugging Face Transformers"
architecture: "BLIP (Bootstrapping Language-Image Pre-training) for VQA"
weights_format: "PyTorch safetensors / bin"
weights_size_bytes: 890000000 (~850 MB)
license: "BSD-3-Clause"
canonical_source: "https://huggingface.co/Salesforce/blip-vqa-base"
paper: "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (ICML 2022)"

hardware_requirements:
  target_device: "mps / cpu"
  minimum_vram_gb: 1.5
  quantization: "none (fp32/fp16)"

input_specification:
  image:
    format: "RGB 3-channel 8-bit or float tensor"
    normalization: "Mean [0.48145466, 0.4578275, 0.40821073], Std [0.26862954, 0.26130258, 0.27577711]"
    target_dimensions: [384, 384]
  text:
    type: "Natural language query string"
    max_tokens: 128

output_specification:
  answer: "Decoded natural language response string"
  confidence: "null (uncalibrated softmax logits are not passed off as calibrated certainty)"
```

### 1.2 Single-Image Text-Guided Grounding (`SINGLE_GROUNDING`)

```yaml
model_id: "google/owlvit-base-patch32"
framework: "PyTorch / Hugging Face Transformers"
architecture: "OWL-ViT (Open-World Localization with Vision Transformers)"
weights_format: "safetensors"
weights_size_bytes: 620000000 (~600 MB)
license: "Apache-2.0"
canonical_source: "https://huggingface.co/google/owlvit-base-patch32"
paper: "Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022)"

hardware_requirements:
  target_device: "mps / cpu"
  minimum_vram_gb: 1.2
  quantization: "none (fp32)"

input_specification:
  image:
    format: "RGB 3-channel 8-bit raster array"
    target_dimensions: [768, 768]
  text_queries:
    type: "List[str] referring expression queries (e.g. ['water body', 'runway', 'building'])"

output_specification:
  boxes: "List of normalized bounding boxes [x1, y1, x2, y2] in [0, 1] range"
  scores: "Sigmoid detection confidence logits per box"
  threshold: 0.10
```

### 1.3 GPU Target Remote-Sensing Adapter (`GeoChatAdapter`)

```yaml
model_id: "MBZUAI/geochat-7b"
framework: "PyTorch / LLaVA-based architecture"
architecture: "GeoChat (Remote Sensing Multimodal Dialogue & Grounding)"
weights_size_bytes: 13500000000 (~13.5 GB)
license: "Apache-2.0"
canonical_source: "https://huggingface.co/MBZUAI/geochat-7b"
paper: "GeoChat: Grounded Large Vision-Language Model for Remote Sensing (CVPR 2024)"
status: "Available via modular registry adapter for GPU compute environments"
```

---

## 2. Provenance Tracking Schema

Every inference response returned by `/api/analysis` encodes an explicit `ModelProvenance` block:

```json
{
  "model_name": "Salesforce/blip-vqa-base",
  "task_family": "SINGLE_VQA",
  "device": "cpu",
  "inference_latency_ms": 1420.5,
  "input_asset_id": "img_b0c18a5274",
  "input_resolution_gsd": 10.0,
  "crs": "EPSG:32644",
  "confidence_calibrated": false,
  "telemetry_steps": [
    {
      "step": "asset_resolution",
      "status": "completed",
      "detail": "Loaded GeoTIFF img_b0c18a5274 (100x100, 3 bands, EPSG:32644)"
    },
    {
      "step": "model_inference",
      "status": "completed",
      "detail": "Executed BLIP VQA on 384x384 tensor"
    }
  ]
}
```

---

## 4. SiamUNet Benchmark Performance Note (Added Prompt 5.5 Audit)

**Date:** 2026-09-09  
**Finding:** The HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff checkpoint achieves
near-zero recall (F1=0.0015) on the ericyu/LEVIRCD_Cropped256 public validation
split at threshold=0.50 with ImageNet normalization preprocessing.

**Root cause (hypothesised):** The HZDR training used an internal dataset split
and possibly different radiometric preprocessing. No public model card specifies
exact normalization. The ericyu HuggingFace dataset may represent a different
train/val partitioning from the HZDR internal benchmark.

**Production impact:** The `analytical_cva` method is the registered default for
bi-temporal change detection. CVA uses image-derived adaptive thresholding and
correctly detects synthetic and real LEVIR-CD changes. The SiamUNet path is
available (weights load correctly, inference executes) but is documented as having
low recall on this specific dataset split at the standard threshold.

**Corrected benchmark:** `docs/TEMPORAL_BENCHMARK_RESULTS_VERIFIED.json`
- n=50 validation pairs, rows 0-49, sequential, no shuffle
- Threshold=0.50, NOT tuned on validation set
- Preprocessing: ImageNet mean/std normalization (matches production inference)
- Precision=0.0212, Recall=0.0008, F1=0.0015, IoU=0.0008

**Prior erroneous summary claim:** Precision=0.72, F1=0.7059 — these numbers
were not present in any saved file and have been retracted.

**Compliance:** Per SatQuery AI Specification requirements, the system uses a "defensible
analytical method" (CVA) as its primary change detection path. The SiamUNet
checkpoint is documented as a "real model trained on LEVIR-CD256" with explicit
performance limitations disclosed in the API response metadata.
