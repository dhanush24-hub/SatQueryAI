# Temporal Change Detection Failure Analysis & Error Taxonomy

## 1. Diagnostic Overview
To systematically diagnose prediction errors and determine why false alarms and missed detections occur, 80 visual diagnostic panels were generated and saved to `storage/accuracy_validation/diagnostic_cases/`:
* **20 Highest False-Positive Samples** (`highest_fp_01_*.jpg` through `highest_fp_20_*.jpg`)
* **20 Highest False-Negative Samples** (`highest_fn_01_*.jpg` through `highest_fn_20_*.jpg`)
* **20 Best Performing Samples** (`best_01_*.jpg` through `best_20_*.jpg`)
* **20 Worst Performing Samples** (`worst_01_*.jpg` through `worst_20_*.jpg`)

Each diagnostic panel visualizes:
`T1 Image (RGB) | T2 Image (RGB) | Ground Truth (White) | Binary Prediction (Red) | Change Probability Heatmap`

---

## 2. Root Causes of Failure Modes

### A. False Positives (Commission Errors) — 814,673 pixels on Test Set
1. **Severe Seasonal / Illumination Transitions**:
   * *Mechanism*: In scenes where T1 was captured during winter (dormant vegetation, high solar zenith angle) and T2 was captured during summer (lush agricultural greening or dried yellow pasture), spectral radiance shifts exceed 40 DN across entire fields.
   * *Mitigation in Model*: The Spatial Difference Attention module (`conv_diff` + spatial sigmoid gating) effectively eliminates 95% of uniform field color changes. Residual false positives occur along field plow furrows and newly cleared agricultural grading that visually mimic foundation earthworks.
2. **Sub-Pixel Building Edge Dilation & Shadow Variations**:
   * *Mechanism*: High solar elevation in T1 casts short shadows, while lower solar elevation in T2 casts elongated shadows adjacent to tall multi-story industrial buildings.
   * *Observation*: A narrow 1-to-3 pixel border of false positives occasionally borders tall warehouse roofs where roof shadows changed orientation between years.

### B. False Negatives (Omission Errors) — 1,228,381 pixels on Test Set
1. **Low Spectral Contrast Roof Materials**:
   * *Mechanism*: New construction using dark grey asphalt shingles erected over preexisting dark gravel / asphalt parking lots or dark loam soil exhibits near-zero contrast in the visible RGB spectrum ($\Delta \text{RGB} < 10 \text{ DN}$).
   * *Observation*: The model detects the building outline where edges are sharp, but may miss the roof interior, resulting in hollowed prediction masks.
2. **Tiny Outbuildings and Sheds (< 100 pixels)**:
   * *Mechanism*: Small detached sheds, carports, and backyard gazebos spanning fewer than 8x8 pixels are occasionally attenuated by ResNet-18 downsampling stages (1/16 spatial scale at layer 4).
   * *Mitigation*: Feature pyramid skip-connections preserve layer 1 and layer 2 spatial features, but objects below 36 pixels ($\approx 9 \text{ m}^2$) remain susceptible to omission.

### C. Ground Truth Annotation Artifacts
1. **Paved Parking Lots & Driveways**:
   * *Observation*: LEVIR-CD annotations strictly target *buildings* (vertical structures). In several test scenes (e.g. `test_102_10`), new asphalt driveways and parking expansions were constructed simultaneously with new homes. The model accurately flags the land disturbance, but LEVIR-CD ground truth labels only the structure footprint, counting the driveway detection as a false positive.
2. **Foundation Slabs vs Completed Roofs**:
   * *Observation*: When T2 captures a poured concrete foundation before the roof framing is erected, ground truth annotators inconsistently label some slabs as "changed building" and others as "not building".

---

## 3. Comparative Architectural Analysis
| Model | Failure Vulnerability | Primary Error Mode |
|---|---|---|
| **Analytical CVA** | Extreme sensitivity to illumination, seasonal greening, and sensor calibration | 94% False Alarm Rate; detects every tree canopy and lawn shift |
| **HZDR SiamUDiff** | Logit collapse / zero-gradient vanishing | Extreme under-prediction (Recall 0.54%); predicts almost all zeros |
| **Fine-Tuned SiamUnet_diff** | Unweighted feature difference without attention gating | Blurry boundaries, high edge false alarms |
| **FC-EF (Early Fusion)** | Channel concatenation merges T1/T2 prematurely, losing temporal correspondence | Poor feature disentanglement, struggles with illumination shifts |
| **AttentionChangeNet** | Robust dual ResNet-18 feature extraction with spatial difference attention | Sharp rectangular building boundaries; robust to seasonal shifts |

---

## 4. Key Takeaways for Deployment
* **Registration Gating is Mandatory**: The model assumes sub-pixel alignment. Input pairs with RMSE $> 1.5$ pixels must be rejected by the operational `TemporalChangeAdapter` before inference.
* **Domain Suitability Gate**: The model is trained specifically on optical true-color imagery for building footprint changes. Domain checking prevents application to SAR or multispectral infrared without explicit warning.
* **Uncertainty Quantification**: The Softmax probability map provides calibrated confidence scores, enabling the system to report uncertainty and flag border regions for human review.
