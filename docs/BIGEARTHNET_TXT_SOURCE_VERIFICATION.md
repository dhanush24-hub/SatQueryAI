# BigEarthNet.txt: Source Verification Report

**Verification Date:** 2026-09-10  
**Evaluator:** SatQuery AI Engineering Audit  
**Compliance Target:** SatQuery AI SatQuery AI Specification / Prompt 8.5

---

## 1. Primary Source Attribution

- **Paper ID:** `arXiv:2603.29630` (v2, revised 2026-04-01)
- **Paper Title:** *"BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation"*
- **Official Authors (Full Names):**
  1. Johann-Ludwig Herzog (TU Berlin / BIFOLD)
  2. Mathis Jürgen Adler (TU Berlin / BIFOLD)
  3. Leonard Hackel (TU Berlin / BIFOLD)
  4. Yan Shu (TU Berlin / BIFOLD)
  5. Angelos Zavras (National Observatory of Athens)
  6. Ioannis Papoutsis (National Observatory of Athens)
  7. Paolo Rota (University of Trento)
  8. Begüm Demir (TU Berlin / BIFOLD)
- **Official Project Website:** [https://txt.bigearth.net](https://txt.bigearth.net)
- **Official Hugging Face Repository:** [`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)
- **Dataset Revision (SHA):** `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`
- **Last Modified on Hub:** 2026-04-01 09:29:12 UTC
- **License:** Community Data License Agreement – Permissive – Version 1.0 (`cdla-permissive-1.0`)

---

## 2. Files Present in Official Repository

The repository contains exactly 10 files:
1. `.gitattributes`
2. `BigEarthNet.txt.parquet` (466,819,745 bytes = ~466.8 MB) — contains all 9,553,962 text annotations.
3. `README.md` — official dataset card and data loading documentation.
4. `ben_txt_datamodule.py` — official PyTorch Dataset (`BENTxTDataset`) and PyTorch Lightning DataModule (`BENTxTDataModule`).
5. `example_data_loading.py` — reference integration examples.
6. `pyproject.toml` — build configuration.
7. `static/images/dataset.svg`
8. `static/images/logos/BIFOLD_Logo_farbig.svg`
9. `static/images/logos/RSiM_Logo.png`
10. `static/images/logos/tu-berlin-logo-long-red.svg`

---

## 3. Dataset Characteristics & Audit against Expected Specifications

| Property | Official Paper Specification | Official Repo Value | Verification Verdict |
|---|---|---|:---:|
| **Total Image Pairs** | 464,044 co-registered S1 + S2 | 464,044 pairs | **MATCH** |
| **Total Text Annotations** | ~9.6 million | **9,553,962 rows** (78 row groups) | **MATCH** |
| **Sentinel-1 Bands** | C-band dual-pol SAR | `VV`, `VH` | **MATCH** |
| **Sentinel-2 Bands** | 12 multispectral channels | `B01`, `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B09`, `B11`, `B12` | **MATCH** |
| **Task Diversity** | Captions, VQA, Grounding | `captioning`, `binary`, `mcq`, `bounding box` | **MATCH** |
| **Benchmark Image Pairs** | 1,082 image pairs | **1,082 image pairs** | **MATCH** |
| **Benchmark Text Annotations** | 15,029 annotations | **15,029 annotations** (`split == "bench"`) | **MATCH** |

---

## 4. Where Actual S1/S2 Imagery Comes From

> [!CRITICAL]
> **The ~467 MB Hugging Face repository `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` contains ONLY the textual annotations/metadata and Python datamodule loaders. It does NOT contain the raw satellite rasters.**

According to the official dataset card and `ben_txt_datamodule.py`:
1. **Raster Origin:** The actual physical Sentinel-1 and Sentinel-2 imagery comes from the **BigEarthNet v2.0 (reBEN)** archive hosted at [https://bigearth.net/](https://bigearth.net/).
2. **DL Storage Format:** The authors prescribe using the `rico-hdl` tool (`github.com/rsim-tu-berlin/rico-hdl`) to convert the downloaded GeoTIFFs into serialized `safetensors` stored within an **LMDB database** (`Encoded-BigEarthNet/`).
3. **Band Access:** `BENImageReader` in `ben_txt_datamodule.py` opens `Encoded-BigEarthNet/` via LMDB using the `patch_id` (for S2) and `s1_name` (for S1) mapped from `BigEarthNet.txt.parquet`.
4. **Volume:** The full archive spans hundreds of gigabytes (12 multispectral bands + 2 SAR bands at 120x120 pixels across 464,044 scenes across Europe).

---

## 5. Band Configurations & Definitions

As defined in `ben_txt_datamodule.py`:
- `_s1_bandnames = ["VV", "VH"]`
- `_s2_bandnames = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]`
- Predefined combinations:
  - `"RGB"`: `["B04", "B03", "B02"]`
  - `"S2-10m20m"`: `["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]` (10 optical channels)
  - `"S1S2-10m20m"`: `["VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]` (12 multi-sensor channels)
  - `"all"`: All 14 channels

---

## 6. Official Splits & Benchmark Semantics

The dataset provides a dedicated categorical `split` column with four values:
- `train`: Official training partition (from BigEarthNet v2.0 train split).
- `validation`: Validation partition for model/checkpoint selection and hyperparameter tuning (from BigEarthNet v2.0 val split).
- `test`: Unverified test partition (remaining portion of BigEarthNet v2.0 test split).
- `bench`: **The curated, human-verified benchmark split (1,082 image pairs, 15,029 annotations)**.

### Precise Relationship Between `bench` and the BigEarthNet v2.0 `test` Partition:
1. **Origin:** In the foundational BigEarthNet v2.0 (reBEN) benchmark, all evaluation tiles belong to a geographic `test` split.
2. **Subsetting & Human Verification:** For BigEarthNet.txt (arXiv:2603.29630), the authors selected a representative subset of **1,082 image pairs** exclusively from this BigEarthNet v2.0 `test` split and conducted rigorous manual human verification on all **15,029 textual annotations** (caption factual accuracy, binary VQA ground truth, MCQ distractors/answers, and referring expression bounding box coordinates).
3. **Parquet Materialization:** In `BigEarthNet.txt.parquet`, these 1,082 verified test image pairs are materialized with `split == "bench"` to distinguish them from unverified test annotations (`split == "test"`).
4. **DataModule Semantics (`ben_txt_datamodule.py`):**
   - General test loader (`stage == "test"`): loads both `splits=['test', 'bench']` (the full BigEarthNet v2.0 test partition).
   - Dedicated benchmark loader (`stage == "bench"`): loads strictly `splits=['bench']` (the 1,082 human-verified pairs).
5. **Leakage Guarantee:** `bench` image pairs and patch IDs are **100% disjoint from `train` and `validation`**. They were never part of training in BigEarthNet v2.0 and must never be used for training, LoRA fitting, threshold selection, or prompt tuning.

---

## 7. Readiness Assessment

- Can `BigEarthNet.txt.parquet` metadata, prompts, and benchmark annotations be loaded? **YES.**
- Can the full hundreds-of-gigabytes raw Sentinel-1 and Sentinel-2 multi-spectral LMDB archive be stored or trained on an 8 GB development laptop? **NO.**
- Full multi-sensor BigEarthNet.txt image-text adaptation is hardware-constrained unless an authentic lightweight patch subset or external GPU compute with `Encoded-BigEarthNet/` is mounted.
