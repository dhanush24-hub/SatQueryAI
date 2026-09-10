# LEVIR-CD Dataset Split Provenance and Disjointness Audit

## 1. Provenance & Source
* **Dataset**: LEVIR-CD (LEVIR Change Detection Dataset)
* **Dataset Paper**: Chen & Shi, *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*, Remote Sensing 2020.
* **Remote Sensing Modality**: High-resolution (0.5m/pixel) bi-temporal true color (RGB) optical satellite/aerial imagery acquired via Google Earth.
* **Geographic Coverage**: 20 different urban/suburban regions and counties across Texas, United States (acquired between 2002 and 2018).
* **Parquet Format**: Extracted from Hugging Face repository `torchgeo/levircd` into local parquet files under `data/levircd/`.

## 2. Split Partitioning & Scene-Level Disjointness
The official LEVIR-CD protocol decomposes large 1024x1024 satellite scenes into 16 non-overlapping 256x256 tiles per scene. To avoid data leakage across adjacent tiles of the same physical geographic scene, train, validation, and test splits MUST partition at the scene level rather than the patch level.

### Split Statistics
| Split | Parquet File | Total Patches (256x256) | Unique Scene IDs | Patches Per Scene |
|---|---|---|---|---|
| **Train** | `data/levircd/train.parquet` | 7,120 | 445 | 16 |
| **Validation** | `data/levircd/val.parquet` | 1,024 | 64 | 16 |
| **Test** | `data/levircd/test.parquet` | 2,048 | 128 | 16 |
| **Total** | — | **10,192** | **637** | **16** |

### Split Overlap & Leakage Audit
A comprehensive set-intersection audit was executed across all scene IDs and patch filenames:
* `train_scenes ∩ val_scenes`: **0** (Disjoint)
* `train_scenes ∩ test_scenes`: **0** (Disjoint)
* `val_scenes ∩ test_scenes`: **0** (Disjoint)
* `train_patches ∩ test_patches`: **0** (Disjoint)
* File hashes and manifest manifests generated at:
  * `data/manifests/levircd_train_manifest.json` (SHA-256 verified)
  * `data/manifests/levircd_val_manifest.json` (SHA-256 verified)
  * `data/manifests/levircd_test_manifest.json` (SHA-256 verified)

## 3. Class Imbalance & Zero-Change Tile Distribution
Because building development is naturally sparse across large geographic regions:
* **Train Split Changed Pixels**: ~4.8% prevalence
* **Val Split Changed Pixels**: ~5.1% prevalence
* **Test Split Changed Pixels**: **5.094%** prevalence (5,609,023 changed pixels / 134,217,728 total pixels)
* **Empty-Change Test Tiles (`gt_sum == 0`)**: **1,113 tiles (54.35%)**
* **Non-Empty Test Tiles (`gt_sum > 0`)**: **935 tiles (45.65%)**

This 54.35% empty-tile prevalence was the central discovery explaining the previous metric discrepancy: tile-averaged macro F1 assigns 0.0 to any tile where a single false-positive occurs on an empty scene, whereas the official remote sensing standard accumulates the global confusion matrix across all 134M test pixels.
