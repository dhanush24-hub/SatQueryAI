# BigEarthNet.txt: Dataset Provenance & Architectural Specification

## 1. Dataset Provenance & Attribution

- **Dataset Name:** BigEarthNet.txt
- **Official Release:** 2026
- **Reference Paper:** *"BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation"* (arXiv:2603.29630)
- **Research Group:** Remote Sensing Image Analysis Group (RSiM), Technische Universität Berlin & Berlin Institute for the Foundations of Learning and Data (BIFOLD)
- **Primary Authors / Curators:** P. Herzog, M. Clasen, B. Demir et al.
- **Repository URL:** [https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)
- **Project Website:** [https://txt.bigearth.net](https://txt.bigearth.net)
- **License:** Community Data License Agreement – Permissive – Version 1.0 ([CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/))

---

## 2. Relationship to BigEarthNet v1.0 and v2.0 (reBEN)

```
BigEarthNet v1.0 (2019)
   │  590,326 S2 patches / 43 CLC classes (noisy labels)
   ▼
BigEarthNet v2.0 / reBEN (2024)
   │  Curated 464,044 co-registered S1+S2 pairs, filtered noise & snow
   ▼
BigEarthNet.txt (2026)
   └── Adds 9,553,962 natural-language annotations (VQA, Referring, Captions)
```

BigEarthNet.txt is **not** a simple collection of image class labels. It is a massive multi-sensor image-text corpus that pairs the physical rasters of reBEN with 9.55 million structured natural language instructions and questions.

---

## 3. Sensor & Physical Modality Specifications

### A. Sentinel-2 (Multispectral Optical)
- **Bands Included (12 channels):**
  - *10m GSD:* B02 (Blue, 490nm), B03 (Green, 560nm), B04 (Red, 665nm), B08 (NIR, 842nm)
  - *20m GSD:* B05 (RedEdge 705nm), B06 (RedEdge 740nm), B07 (RedEdge 783nm), B8A (Narrow NIR 865nm), B11 (SWIR-1 1610nm), B12 (SWIR-2 2190nm)
  - *60m GSD:* B01 (Coastal Aerosol 443nm), B09 (Water Vapor 945nm)
- **Standard Predefined Subsets:**
  - `RGB`: [B04, B03, B02]
  - `S2-10m20m`: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12] (10 spectral channels)
- **Patch Dimension:** Resampled to $120 \times 120$ pixels ($1.2\text{ km} \times 1.2\text{ km}$ footprint)
- **Radiometric Values:** Bottom-of-Atmosphere (L2A) surface reflectance (Digital Numbers)

### B. Sentinel-1 (Synthetic Aperture Radar / SAR)
- **Instrument:** C-band Synthetic Aperture Radar (5.405 GHz)
- **Product Type:** Level-1 Ground Range Detected (GRD), Interferometric Wide Swath (IW)
- **Polarizations (2 channels):**
  - `VV`: Co-polarized vertical-transmit, vertical-receive
  - `VH`: Cross-polarized vertical-transmit, horizontal-receive
- **Radiometric Unit:** Calibrated backscatter coefficient $\sigma^0$ in decibels (dB)
  - Official Train Means: $\text{VV} = -12.64\text{ dB}$, $\text{VH} = -19.35\text{ dB}$
  - Official Train Stds: $\text{VV} = 5.13\text{ dB}$, $\text{VH} = 5.59\text{ dB}$

---

## 4. Annotation Schema & Split Distribution

### A. Dataset Totals
- **Total Image Pairs:** 464,044 Sentinel-1 + Sentinel-2 tiles
- **Total Text Annotations:** 9,553,962 rows (stored in 78 Parquet row groups)

### B. Split Hierarchy
1. `train` (~40.4%): For model optimization.
2. `validation` (~25.9%): Strictly for hyperparameter tuning and model selection.
3. `test` (~33.5%): Standard test set.
4. `bench` (~0.15%, ~14,000 annotations): **Curated, human-verified benchmark split** reserved exclusively for final evaluation. **Never included in training.**

### C. Task Taxonomy
| Task Type (`type`) | Categories (`category`) | Description | Example Target Output |
|---|---|---|---|
| `binary` | `presence`, `area`, `count`, `adjacency` | Binary verification of LULC features | `"yes"`, `"no"` |
| `mcq` | `climate zone`, `season`, `country`, `area` | Multiple choice QA with 4 candidate options | Option letter or text |
| `bounding box` | `point`, `reference` | Referring expression spatial grounding | Bounding box coordinates $[x_1, y_1, x_2, y_2]$ |
| `captioning` | `LULC summary` | Comprehensive scene description | Dense natural-language paragraph |
