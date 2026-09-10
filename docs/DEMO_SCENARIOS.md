# SatQuery AI: SIH Jury Demonstration Script & Scenarios

This document provides a guided walkthrough for demonstrating SatQuery AI to the Smart India Hackathon evaluation panel.

---

## 1. Demo Data Package & Provenance

All demonstration assets are sourced from verified open-access remote sensing repositories with permissive licensing:

| Scene Name | Source Dataset | License | Modality | Ground Truth / Phenomena |
|---|---|---|---|---|
| `demo_single_urban.tif` | OpenAerialMap / SpaceNet | CC-BY 4.0 | Optical RGB (0.5m GSD) | High-density residential settlement |
| `demo_levir_t1.tif` | LEVIR-CD Held-Out Test Set | Open Access / Non-Commercial Research | Optical RGB (0.5m GSD) | Pre-construction baseline (2018) |
| `demo_levir_t2.tif` | LEVIR-CD Held-Out Test Set | Open Access / Non-Commercial Research | Optical RGB (0.5m GSD) | Post-construction expansion (2022) |
| `demo_multisensor_opt.tif` | SEN12MS / Sentinel-2 | CC-BY-SA 4.0 | Optical RGB (10m GSD) | Riverine delta with cloud occlusion |
| `demo_multisensor_sar.tif` | SEN12MS / Sentinel-1 GRD | CC-BY-SA 4.0 | SAR VV/VH Calibrated $\sigma^0$ dB | Specular water body (penetrates clouds) |

---

## 2. Step-by-Step Jury Demonstration Script

### Scenario 1: Natural-Language Bi-Temporal Change Detection
- **Narrative:** "The analyst uploads two optical satellite images taken 4 years apart and asks a natural language question about urban growth."
- **User Action:**
  1. Open `http://localhost:3000`.
  2. Drag and drop `demo_levir_t1.tif` and `demo_levir_t2.tif`.
  3. Enter Query: *"Where did new structures appear between these two dates?"*
  4. Click **Run Analysis**.
- **System Behavior:**
  - System automatically identifies 2 optical inputs with temporal separation.
  - Verifies registration: Zero-mean Normalized Cross-Correlation $> 0.85$ (Status: HIGH).
  - Invokes `AttentionChangeNet` ($\tau = 0.50$, held-out test $F_1 = 0.8459$).
  - Displays side-by-side synchronized viewer and interactive swipe slider.
  - Highlights clustered changes with bounding boxes and polygon overlays.
  - Exports a professional 2-page PDF Dossier and standard GeoJSON file.

---

### Scenario 2: Complementary Optical + SAR Cross-Modal Intelligence
- **Narrative:** "Optical imagery is partially obscured by cloud shadows, confusing standard detectors. We add calibrated Sentinel-1 radar to uncover ground reality."
- **User Action:**
  1. Upload `demo_multisensor_opt.tif` (Optical) and `demo_multisensor_sar.tif` (SAR).
  2. Enter Query: *"Extract the true water extent across optical and radar sensors."*
  3. Click **Run Analysis**.
- **System Behavior:**
  - System detects 1 Optical + 1 SAR image; automatically routes to `OPTICAL_SAR_ANALYSIS`.
  - Disambiguates SAR representation: recognizes calibrated dB backscatter without touching raw disk files.
  - Applies in-memory Enhanced Lee speckle filter.
  - Performs cross-modal alignment check via Normalized Mutual Information (NMI).
  - Surfaces **Cross-Modal Agreement** where both sensors corroborate water.
  - Surfaces **Cross-Modal Disagreement** where optical shadow was flagged, explaining that SAR backscatter ruled out water.
  - `confidence` is left strictly as `null` with scientific honesty.

---

### Scenario 3: Automated Safeguard on Unsupported Conversion Queries
- **Narrative:** "An analyst asks whether agricultural land was converted to forest. SatQuery AI refuses to hallucinate multiclass transitions because the model is binary-trained."
- **User Action:**
  1. Upload `demo_levir_t1.tif` and `demo_levir_t2.tif`.
  2. Enter Query: *"Was forest converted to commercial buildings here?"*
  3. Click **Run Analysis**.
- **System Behavior:**
  - Model detects semantic transition query words ("converted to", "forest to buildings").
  - Safeguard triggers: explicitly informs analyst that the change detector only localizes structural boundaries and cannot infer multiclass land-cover transitions without multiclass labels.
  - Status returned: `SUPPORTED_WITH_WARNINGS`.

---

### Scenario 4: Strict Refusal of Unreferenced Geolocation
- **Narrative:** "When given ordinary unreferenced rasters (e.g. standard PNG/JPEG screenshots), SatQuery refuses to invent fake GPS coordinates or fake square meters."
- **User Action:**
  1. Upload two unreferenced screenshot files (`test.png`).
  2. Enter Query: *"Calculate the exact square kilometers of new roads."*
- **System Behavior:**
  - System detects missing Coordinate Reference System (`crs = None`).
  - Analysis proceeds on pixel grid, but physical area display rules suppress `area_m2` and `area_ha`.
  - GeoJSON export returns HTTP 400 with explanation: *"Cannot generate geographic GeoJSON for unreferenced pixel grid."*
