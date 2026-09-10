# SatQuery AI — Current State Prototype Audit
**Project:** SatQuery AI (Team Alpha Logic — SIH 2026 / Problem Statement 26167)  
**Date:** September 2026  
**Audit Scope:** Repository Architecture, Frontend, Backend, APIs, Data Handling, and Model Implementations

---

## 1. Executive Summary

The existing repository is an early-stage research prototype built with **Next.js 15 / React 19 / Vinext** (with Cloudflare Workers packaging capabilities). It provides an interactive mission workspace with image inspection, question composition, and a multi-step evidence gating pipeline.

However, the prototype currently lacks a dedicated Python/FastAPI remote sensing backend, lacks genuine geospatial raster ingestion (GDAL/GeoTIFF parsing), and contains several dangerous mock layers:
1. **Synthetic Telemetry and Metrics on User Uploads**: `lib/isro-data.ts` dynamically generates fabricated satellite platform data (e.g. claiming user images are from "ISRO EOS-04 / Cartosat-3"), synthetic LULC delta percentages, fictional spectral indices (NDVI, NDWI, NDBI), and hardcoded 88% confidence scores.
2. **Hardcoded Hectare Calculations**: `components/mission-findings.tsx` and `lib/analysis.ts` calculate fictional surface area deltas (`+4.2 ha`, `+14.5 ha`, `-2.8 ha`) using crude keyword string matching on finding categories.
3. **Simulated Spectral Lenses**: `components/imagery-canvas.tsx` implements "CIR", "NDVI", "NDWI", and "SAR" views using CSS filter color transforms (`contrast(190%) hue-rotate(65deg)`) rather than actual multi-spectral band math.
4. **General LLM vs. Specialist RS Models**: The "live" pipeline uses client-side or server-proxied Google Gemini API calls running custom prompts rather than specialized, remote-sensing fine-tuned vision-language models or grounding models.

---

## 2. Technical Stack Audit

| Dimension | Current Implementation | Target Production State | Status |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | Next.js 15 (React 19, TypeScript, Tailwind CSS, Lucide icons, Base-UI / Shadcn primitives) running on Vinext / Vite | Preserved Next.js frontend with unified typed API client layer | `WORKING` |
| **Backend Framework** | Next.js API Routes (`app/api/plan`, `app/api/analyze`) running on Node.js / edge runtime | Dedicated Python FastAPI backend (`backend/app`) | `MISSING` |
| **Database & Persistence** | In-memory React state (`useState`); no durable database | Relational metadata store (PostgreSQL/SQLite) + Session state | `MISSING` |
| **Storage Layer** | Base64 strings held in memory and JSON request bodies; size capped at 10 MB | Dedicated Storage Service (`LocalStorageService` / S3) storing raw GeoTIFF and benchmark PNG/JPEG | `MISSING` |
| **Authentication** | None | Role-based authentication / API keys (out of scope for initial prototype) | `MISSING` |
| **Geospatial Processing** | Browser `Image.decode()` on PNG/JPEG; no CRS, projection, or band handling | Python Geospatial Service (GDAL / Rasterio / pyproj / numpy) for GeoTIFF & SAR | `MISSING` |
| **Model Registry** | Hardcoded TypeScript map `CapabilityRegistry` in `lib/registry.ts` delegating to Gemini prompts | Modular Python Model & Tool Registry with specialist RS models | `PARTIAL` |
| **Evidence Gate** | Deterministic validation logic in `lib/validation.ts` checking ROI bounds and counterpart evidence | Preserved and integrated into orchestrator evidence gate service | `WORKING` |

---

## 3. Component Classification (WORKING / PARTIAL / MOCK / MISSING)

### 3.1. Frontend & UI Components

- **Landing Page (`app/page.tsx`)** — `WORKING`
  - Explains problem statement, multi-modal capabilities (Observe, Compare, Fuse as automatic internal mechanisms), and links to workspace. Clean, responsive, and adheres to design system.
- **Mission Workspace (`app/workspace/page.tsx`)** — `WORKING` (with partial mock bindings)
  - Interactive multi-panel interface: observation canvas, conversation thread, findings accordion, ISRO analytics panel, workflow execution viewer, and limitation notices.
- **Observation Canvas (`components/imagery-canvas.tsx`)** — `PARTIAL` / `MOCK`
  - `WORKING`: Pan/zoom, image carousel, side-by-side mode, swipe slider, flicker comparison, synchronized crosshairs, and bounding box rendering.
  - `MOCK`: Spectral filter lenses (CIR, NDVI, NDWI, SAR) are pure CSS filter manipulations (`contrast`, `hue-rotate`, `sepia`). These must be isolated or marked as visual display simulations, not true radiometric band operations.
- **Mission Findings Panel (`components/mission-findings.tsx`)** — `PARTIAL`
  - `WORKING`: Verdict indicators (`supported`, `partially_supported`, `conflicting`), candidate finding statements, alternative explanations, and visual grounding cross-highlighting.
  - `MOCK`: Hardcoded hectare badges computed via string regex (`+4.2 ha` for urban, `+14.5 ha` for water).
- **ISRO Analytics Component (`components/isro-analytics.tsx`)** — `MOCK`
  - Displays polished telemetry cards, LULC change bar charts, spectral indices, and a gauge for `{confidenceScore}% GATED CONFIDENCE`.
  - Claims "ISRO NRSC SPEC COMPLIANT" even when user uploads arbitrary images. All underlying data is synthesized in `lib/isro-data.ts`.

---

### 3.2. Data Processing & API Layer

- **Image Ingestion & Uploads (`app/workspace/page.tsx`, `lib/api-input.ts`)** — `PARTIAL`
  - `WORKING`: File type validation (`image/png`, `image/jpeg`, `image/webp`), size limits (5 MB/file, 10 MB total), dimension verification, and duplicate base64 hashing.
  - `MISSING`: GeoTIFF (`.tif`, `.tiff`) decoding, band separation, CRS extraction, resolution/GSD calculation, and local file storage.
- **Workflow Planning Route (`app/api/plan/route.ts`)** — `PARTIAL`
  - `WORKING`: Input assessment and validation logic (`lib/workflow.ts`). Identifies whether question requires single-scene, temporal, or optical-SAR reasoning.
  - `MOCK`: Relies on Gemini LLM with prompt constraints to output a planned JSON structure.
- **Analysis Execution Route (`app/api/analyze/route.ts`)** — `PARTIAL`
  - `WORKING`: NDJSON streaming response protocol emitting real-time execution steps and final synthesis.
  - `MOCK`: Executes Gemini prompt chains instead of specialist remote sensing models. When Gemini is not connected, fails or falls back to demo fixtures.

---

### 3.3. Core Logic & Validation

- **Deterministic Evidence Gate (`lib/validation.ts`)** — `WORKING`
  - Rigorous algorithmic verification:
    - Validates that bounding boxes are strictly normalized (0–100%) and valid.
    - Rejects untracked findings introduced in review stages.
    - Requires temporal counterparts: any change finding across two dates must have grounding in both acquisitions.
    - Enforces conflicting verdicts when optical and SAR signals disagree.
    - Forces synthesis to qualify or abstain when evidence is insufficient.
  - *Must be preserved as a core strength of the architecture.*
- **Demonstration Missions (`lib/demos.ts`)** — `WORKING` (as curated fixtures)
  - Curated, historically sourced satellite pairs:
    1. Cairo urban expansion (Landsat 5 & Landsat 8)
    2. Florence flood extent (Landsat 8 & Sentinel-1 ARIA proxy)
    3. Sundarbans mangrove delta (Landsat 7 mosaic)
  - Correctly labeled as curated demonstrations in metadata.
- **Synthetic ISRO Data Generator (`lib/isro-data.ts`)** — `MOCK` (Dangerous)
  - Injects simulated telemetry, LULC deltas, and hardcoded `confidenceScore: 88` into live user uploads.
  - Must be strictly isolated to demo mode and never presented as genuine live inference.

---

### 3.4. Backend & Remote Sensing Infrastructure

- **FastAPI Application (`backend/`)** — `MISSING`
  - Currently no Python service exists. Must be implemented from the ground up with modular routers, Pydantic schemas, and config.
- **GeoTIFF / Raster Ingestion Pipeline** — `MISSING`
  - No GDAL, rasterio, or geospatial metadata extraction tools are currently wired up.
- **Specialist Vision-Language Models (VLM / VQA / Grounding)** — `MISSING`
  - No remote-sensing specialist models (e.g. RemoteCLIP, GeoChat, EarthGPT, or BigEarthNet-adapted models) are integrated yet.
- **Task Orchestrator & Observable Trace Service** — `PARTIAL`
  - TypeScript workflow planner exists; needs Python backend equivalent adhering to observable execution traces (no hidden CoT).

---

## 4. Summary Matrix

| Component / Feature | Classification | Action Required |
| :--- | :--- | :--- |
| Next.js Workspace UI | `WORKING` | Preserve; wire to backend API client |
| Evidence Gate (`validation.ts`) | `WORKING` | Preserve; port contracts to backend |
| Demo Missions (`demos.ts`) | `WORKING` | Preserve; isolate from production path |
| Upload Validation (PNG/JPEG) | `PARTIAL` | Expand to GeoTIFF/TIFF in backend |
| Frontend API Layer | `MISSING` | Create modular client in `lib/api/` |
| FastAPI Backend (`backend/app`) | `MISSING` | Build from scratch |
| Storage Service (`backend/storage`) | `MISSING` | Build local filesystem storage service |
| Model/Tool Registry | `MISSING` | Implement Python registry foundation |
| Domain Schemas (ImageAsset, Analysis) | `MISSING` | Implement Pydantic domain models |
| ISRO Telemetry on User Uploads | `MOCK` | Quarantine to demo; return None for raw uploads |
| Hardcoded Hectares (`+4.2 ha`) | `MOCK` | Remove from user upload display |
| CSS Spectral Filters | `MOCK` | Label as visual approximations |
| Specialist RS Models | `MISSING` | Prepare registry; return `NOT_IMPLEMENTED` |

---

## 5. Architectural Preservation Directives

1. **Keep the UI/UX Intact**: The dual-view observation canvas, interactive bounding boxes, and findings panel are high quality and intuitive.
2. **Keep the No-Mode-Selection Principle**: The user must never manually toggle "Observe", "Compare", or "Fuse". The system interprets the natural language query and input configuration automatically.
3. **No Fake Results**: In the new FastAPI backend, any model endpoint not yet integrated must return an explicit `NOT_IMPLEMENTED` status. Never fabricate satellite findings or confidence values.
