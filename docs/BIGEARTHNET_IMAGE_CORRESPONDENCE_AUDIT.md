# BigEarthNet Image Correspondence Audit Report

**Audit Date:** 2026-09-10  
**Evaluator:** SatQuery AI Engineering Audit  
**Compliance Target:** Smart India Hackathon PS 26167 / Prompt 8.5B  

---

## 1. Executive Summary

A rigorous mathematical and geospatial audit of the Stage A sparse image acquisition pipeline revealed a **critical coordinate transposition bug**:
- In the official BigEarthNet convention (`<product_name>_<H-Order>_<V-Order>`):
  - `H-Order` (Horizontal Order) represents the **column index (X-axis)** starting from 0 at the western edge of the tile.
  - `V-Order` (Vertical Order) represents the **row index (Y-axis)** starting from 0 at the northern edge of the tile.
- In Stage A's `acquire_stageA_dataset.py`, `parts[6]` (`H-Order`) was erroneously assigned to the `row` variable, and `parts[7]` (`V-Order`) was assigned to `col`. The rasterio extraction window was instantiated as:
  ```python
  Window(col_off=c_idx * 120, row_off=r_idx * 120, width=120, height=120)
  ```
  resulting in:
  ```python
  col_off = V_Order * 120  # Incorrect (Vertical used as Column)
  row_off = H_Order * 120  # Incorrect (Horizontal used as Row)
  ```
- **Consequence:** 100% of image patches in Stage A were extracted across the tile diagonal (`col` $\leftrightarrow$ `row`), displacing the imagery between **10.18 km and 147.64 km** from the ground-truth patch footprint.
- **Scientific Impact:** The Vision-Language Model in Stage A was supervised with textual questions/answers describing a patch at `(col=H, row=V)` while being shown satellite imagery from `(col=V, row=H)`. This directly explains the severe negative ("no") collapse and validates Prompt 8.5B's mandate to invalidate Stage A image conclusions and rectify image extraction.

---

## 2. Geospatial Audit of 20 Sample Patches

The table below documents 20 randomly sampled patches from the Stage A train and validation sets, comparing their intended Sentinel-2 L2A footprints against the extracted Stage A footprints:

| # | Patch ID | MGRS Tile | H-Order (Col) | V-Order (Row) | Correct UTM Bounds (minX, maxY, maxX, minY) | Stage A Erroneous Bounds | Spatial Displacement |
|---|---|:---:|:---:|:---:|---|---|:---:|
| 1 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_01_76` | `35VNK` | 1 | 76 | `[501180, 6808840, 502380, 6807640]` | `[591180, 6898840, 592380, 6897640]` | **127.28 km** |
| 2 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_28` | `35VNK` | 0 | 28 | `[499980, 6866440, 501180, 6865240]` | `[533580, 6900040, 534780, 6898840]` | **47.52 km** |
| 3 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_06` | `35VNK` | 0 | 6 | `[499980, 6892840, 501180, 6891640]` | `[507180, 6900040, 508380, 6898840]` | **10.18 km** |
| 4 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_02_13` | `35VNK` | 2 | 13 | `[502380, 6884440, 503580, 6883240]` | `[515580, 6897640, 516780, 6896440]` | **18.67 km** |
| 5 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_70` | `35VNK` | 0 | 70 | `[499980, 6816040, 501180, 6814840]` | `[583980, 6900040, 585180, 6898840]` | **118.79 km** |
| 6 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_62` | `35VNK` | 0 | 62 | `[499980, 6825640, 501180, 6824440]` | `[574380, 6900040, 575580, 6898840]` | **105.22 km** |
| 7 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_57` | `35VNK` | 0 | 57 | `[499980, 6831640, 501180, 6830440]` | `[568380, 6900040, 569580, 6898840]` | **96.73 km** |
| 8 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_35` | `35VNK` | 0 | 35 | `[499980, 6858040, 501180, 6856840]` | `[541980, 6900040, 543180, 6898840]` | **59.40 km** |
| 9 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_02_12` | `35VNK` | 2 | 12 | `[502380, 6885640, 503580, 6884440]` | `[514380, 6897640, 515580, 6896440]` | **16.97 km** |
| 10 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_26` | `35VNK` | 0 | 26 | `[499980, 6868840, 501180, 6867640]` | `[531180, 6900040, 532380, 6898840]` | **44.12 km** |
| 11 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_01_88` | `35VNK` | 1 | 88 | `[501180, 6794440, 502380, 6793240]` | `[605580, 6898840, 606780, 6897640]` | **147.64 km** |
| 12 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNL_13_41` | `35VNL` | 13 | 41 | `[515580, 6950840, 516780, 6949640]` | `[549180, 6984440, 550380, 6983240]` | **47.52 km** |
| 13 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_01_52` | `35VNK` | 1 | 52 | `[501180, 6837640, 502380, 6836440]` | `[562380, 6898840, 563580, 6897640]` | **86.55 km** |
| 14 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_22` | `35VNK` | 0 | 22 | `[499980, 6873640, 501180, 6872440]` | `[526380, 6900040, 527580, 6898840]` | **37.34 km** |
| 15 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_01_64` | `35VNK` | 1 | 64 | `[501180, 6823240, 502380, 6822040]` | `[576780, 6898840, 577980, 6897640]` | **106.91 km** |
| 16 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_01_21` | `35VNK` | 1 | 21 | `[501180, 6874840, 502380, 6873640]` | `[525180, 6898840, 526380, 6897640]` | **33.94 km** |
| 17 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_08` | `35VNK` | 0 | 8 | `[499980, 6890440, 501180, 6889240]` | `[509580, 6900040, 510780, 6898840]` | **13.58 km** |
| 18 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_07` | `35VNK` | 0 | 7 | `[499980, 6891640, 501180, 6890440]` | `[508380, 6900040, 509580, 6898840]` | **11.88 km** |
| 19 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_23` | `35VNK` | 0 | 23 | `[499980, 6872440, 501180, 6871240]` | `[527580, 6900040, 528780, 6898840]` | **39.03 km** |
| 20 | `S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK_00_55` | `35VNK` | 0 | 55 | `[499980, 6834040, 501180, 6832840]` | `[565980, 6900040, 567180, 6898840]` | **93.34 km** |

---

## 3. Rectification Protocol

To ensure 100% authentic spatial correspondence:
1. **Window Definition Fixed:**
   ```python
   # Corrected Window indexing
   col_offset = h_order * 120  # Horizontal index -> column offset
   row_offset = v_order * 120  # Vertical index -> row offset
   window = Window(col_off=col_offset, row_off=row_offset, width=120, height=120)
   ```
2. **All Image Caches Cleared:**
   All transposed Stage A rasters in `data/bigearthnet/rasters/` will be removed and re-extracted with exact `(col=H, row=V)` mapping.
3. **Stage A Conclusion Invalidation:**
   All prior image-dependent benchmark numbers are formally marked as corrupted by the transposition bug and superseded by Stage 8.5B.
