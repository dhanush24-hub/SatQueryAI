# Final Model Provenance and Architecture Specification

**SatQuery AI — System Specification**

## Production Change Detection Model

* **Selected Winner**: `AttentionChangeNet` (Simple Attention Change Network - SACN)
* **Architecture Class**: Dual-stream Siamese Encoder + Difference Attention + Multi-scale Decoder
* **Checkpoint File**: `models/satquery_change_v1/attention_best.safetensors`
* **File Size**: 50,310,920 bytes (~50.3 MB)
* **Total Parameters**: 12,562,474
* **Encoder Backbone**: ResNet-18 (pretrained on ImageNet via `timm.models.resnet18.a1_in1k`)
* **Decoder**: Multi-scale spatial difference attention blocks at strides (4, 8, 16, 32)
* **Inference Device**: Apple Silicon MPS (`torch.backends.mps`) / CPU
* **Loss Function**: Combined NLLLoss (weighted for class imbalance) + DiceLoss

---

## Preprocessing & Normalization Protocol

* **Input Arrays**: T1 and T2 raster arrays in RGB order
* **Pixel Value Scaling**: Raw [0, 255] float array divided by `255.0` to [0.0, 1.0]
* **Standardization**: Standard ImageNet mean/std normalization per channel:
  * Mean: `[0.485, 0.456, 0.406]`
  * Std: `[0.229, 0.224, 0.225]`
* **Inference Tiling**: 256×256 sliding window with 32-pixel overlap and edge reflection padding
* **Calibrated Operating Threshold**: `0.40` (empirically calibrated on held-out validation set)

---

## Alternate Evaluated Models (Retained for Benchmarking)

1. **Fine-Tuned SiamUnet_diff**:
   * Checkpoint: `models/satquery_change_v1/siamunet_diff_best.safetensors` (5.4 MB, 1.35M params)
   * Preprocessing: `/255.0` (Daudt et al. ICIP 2018 protocol)
   * Calibrated Threshold: `0.20`
2. **FC-EF (Early Fusion U-Net)**:
   * Checkpoint: `models/satquery_change_v1/fc_ef_best.safetensors` (7.8 MB, 1.94M params)
   * Calibrated Threshold: `0.40`
3. **Analytical CVA (Change Vector Analysis)**:
   * Parameter-free Euclidean difference magnitude with adaptive 2-sigma thresholding
4. **HZDR SiamUDiff (Pretrained Checkpoint)**:
   * Deprecated / Degenerate baseline: saturates at class 0 logits.
