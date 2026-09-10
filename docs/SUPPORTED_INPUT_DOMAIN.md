# Supported Input Domain Specification

**SatQuery AI — Problem Statement 26167**

This document establishes the verified operating envelope for SatQuery's optical satellite image analysis and change intelligence pipeline.

---

## 1. Operating Domain Matrix

| Property | Supported Optimal | Supported with Warnings | Out-of-Domain / Unsupported |
|---|---|---|---|
| **Modality** | Optical RGB (Aerial/Satellite) | Multispectral (first 3 bands used), Panchromatic (triplicated) | SAR / Radar imagery (Sentinel-1, NISAR SAR) |
| **Bands** | 3 bands (Red, Green, Blue) | 1 band, 4+ bands | Synthetic non-imagery rasters |
| **Ground Sampling Distance (GSD)** | 0.1m — 5.0m / pixel | 5.0m — 30.0m / pixel | > 30.0m / pixel or < 0.05m / pixel |
| **Dimensions** | 256×256 to 4096×4096 pixels | 64×64 to 256×256 pixels | < 64×64 pixels (Insufficient Evidence) |
| **Radiometric Depth** | 8-bit [0, 255] | 16-bit unsigned (auto-stretched) | Complex numbers, floating point non-normalized |
| **Spatial Alignment** | Pre-aligned or common CRS | Reprojection required | Coordinate reference system missing or incompatible |
| **Registration Quality** | Subpixel translation (< 1.5 px) | 1.5 px — 3.0 px translation | Translation > 3.0 px or Correlation < 0.4 |
| **Cloud / Occlusion** | < 10% nodata/cloud | 10% — 60% nodata/cloud | > 60% nodata/cloud (Insufficient Evidence) |

---

## 2. Input Domain Gate Statuses

1. **`SUPPORTED`**: The image meets all optimal criteria. Full pixel-level localization and change clustering are executed with maximum confidence.
2. **`SUPPORTED_WITH_WARNINGS`**: Imagery is processed, but explicit warnings are attached to the output explaining potential sources of error (e.g., panchromatic replication, coarse resolution).
3. **`OUT_OF_DOMAIN`**: Imagery modality or type is unsupported. Execution is safely rejected with clear diagnostic explanations rather than generating fabricated results.
4. **`INSUFFICIENT_EVIDENCE`**: Imagery exhibits severe degradation, co-registration failure, or extensive cloud cover (>60%). The system responds with an evidence limitation verdict.
