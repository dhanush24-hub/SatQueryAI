# SatQuery AI

**AI-Powered Satellite Image Analysis System**

SatQuery AI is an intelligent satellite image analysis system that converts natural-language questions about satellite imagery into structured, evidence-gated geospatial intelligence. It targets earth-observation workflows for flood mapping, urban change detection, land-cover classification, and multi-sensor optical+SAR fusion.


---

## Overview

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, Tailwind CSS, shadcn/ui |
| Backend | FastAPI (Python 3.12), SQLite, Rasterio, PyTorch |
| Orchestration | Multi-step planner → evidence gate → synthesis |
| Temporal Change | AttentionChangeNet, FC-EF, SiamUNet-Diff (LEVIR-CD) |
| Optical+SAR Fusion | Cross-modal alignment + disagreement scoring |
| Multi-label Classification | Experimental BigEarthNet.txt (Stage 8.5B/8.6 — ongoing) |

---

## Key Capabilities

- **Natural-language querying** — no mode-select; intent is automatically classified
- **Temporal change detection** — pixel-level change maps with confidence and uncertainty
- **Optical + SAR fusion** — agreement/disagreement scoring across sensor modalities
- **Evidence gate** — every finding is classified as supported / partially-supported / insufficient / conflicting before synthesis
- **Grounding** — approximate bounding regions linked to findings (not segmentation)
- **PDF report generation** — self-contained analysis reports
- **Demo missions** — Cairo, Florence, Sundarbans running through the live evidence pipeline

---

## Architecture

```
User Query + Images
        │
        ▼
┌─────────────────────┐
│   Intent Classifier  │  (task_classifier.py)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Planner / Executor │  (planner.py, executor.py)
│  (multi-step plan)   │
└────────┬────────────┘
         │
   ┌─────┴──────────────────────┐
   │ Specialist Services         │
   │  • Geospatial preprocessing │  (rasterio, alignment)
   │  • Temporal change models   │  (AttentionChangeNet, FC-EF, SiamUNet)
   │  • Optical/SAR fusion       │  (optical_sar_fusion.py)
   │  • VQA adapter              │  (vqa_adapter.py)
   │  • Visual grounding         │  (grounding_adapter.py)
   └─────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Evidence Assessor  │  (evidence_assessor.py)
│  gate: supported /   │
│  partial / conflict  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Result Integrator   │  → JSON response + PDF report
└─────────────────────┘
```

Full architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Current Implemented Scope

| Feature | Status |
|---------|--------|
| FastAPI backend with health/upload/analysis/jobs endpoints | ✅ Implemented |
| SQLite analysis persistence and history | ✅ Implemented |
| Temporal change detection (AttentionChangeNet) | ✅ Implemented + benchmarked |
| Optical+SAR fusion with cross-modal alignment | ✅ Implemented |
| Evidence gate with verdict classification | ✅ Implemented |
| PDF report generation | ✅ Implemented |
| Next.js frontend — workspace + mission UI | ✅ Implemented |
| Demo missions (Cairo, Florence, Sundarbans) | ✅ Implemented |
| BigEarthNet multi-label classification (Stage 8.5B) | ⚠️ Experimental / ongoing |
| Prompt 8.6 RS-VQA fine-tuning | 🔄 Training in progress — results pending |

> **Scientific Honesty Note:** Prompt 8.6 training results are not yet finalized and are **not** reported here. Stage 8.5B BigEarthNet model performance remains experimental. Only the temporal change detection model (AttentionChangeNet, benchmarked on LEVIR-CD) has published held-out metrics — see [`docs/FINAL_HELDOUT_METRICS.json`](docs/FINAL_HELDOUT_METRICS.json).

---

## Repository Structure

```
satquery/
├── app/                        # Next.js App Router pages and API routes
│   ├── api/                    # Next.js server-side API handlers
│   └── workspace/              # Main workspace page
├── components/                 # React UI components
│   └── ui/                     # shadcn/ui primitives
├── lib/                        # Frontend logic and prompts
│   ├── prompts/                # Modular analysis prompt templates
│   └── api/                    # Backend API client
├── hooks/                      # React hooks
├── public/                     # Static assets and demo mission images
│   └── missions/               # Demo fixture images + PROVENANCE.md
│
├── backend/                    # FastAPI Python backend
│   ├── app/
│   │   ├── api/routes/         # HTTP endpoints (analysis, health, jobs, uploads)
│   │   ├── core/               # Config (pydantic-settings, env-driven)
│   │   ├── db/                 # SQLite session and repositories
│   │   ├── domain/             # Task and modality enums
│   │   ├── registry/           # Model registry
│   │   ├── schemas/            # Pydantic request/response models
│   │   └── services/
│   │       ├── geospatial/     # Rasterio preprocessing, SAR, alignment
│   │       ├── models/         # AttentionChangeNet, FC-EF, SiamUNet, VQA
│   │       ├── orchestration/  # Planner, executor, evidence gate
│   │       ├── reporting/      # PDF generation
│   │       └── storage/        # File storage abstraction
│   ├── tests/                  # pytest test suite (86+ tests)
│   └── requirements.txt
│
├── scripts/                    # Training, evaluation, audit scripts
├── data/
│   ├── manifests/              # Dataset split manifests (JSON, versioned)
│   └── levircd/                # LEVIR-CD parquet splits (git-ignored, see below)
├── models/
│   └── satquery_change_v1/     # Trained change detection model metadata
│       ├── attention_best.*    # Weights (git-ignored, download instructions below)
│       └── provenance.json
├── docs/                       # Scientific reports, validation, architecture
├── storage/
│   ├── fixtures/               # LEVIR demo sample images
│   ├── scenario_*.tif          # Geospatial test fixtures
│   └── test_*.tif / *.png      # Backend test fixtures
├── tests/                      # Frontend unit tests
│
├── Dockerfile
├── docker-compose.yml
├── .env.example                # Environment template (copy to .env)
├── next.config.ts
├── package.json
└── requirements.txt → backend/requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.12+
- Node.js 20+
- `git`

### 1. Clone

```bash
git clone https://github.com/<your-org>/satquery.git
cd satquery
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env:
#   HF_TOKEN=<your huggingface token>
#   GOOGLE_GEMINI_API_KEY=<your gemini key>
#   DEVICE=cpu  # or cuda / mps
```

### 3. Backend

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt

# Start backend
PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"satquery-api","version":"0.1.0"}
```

Run tests:
```bash
cd backend
PYTHONPATH=. pytest tests/ -v
```

### 4. Frontend

```bash
npm install
npm run dev
# → http://localhost:3000
```

TypeScript check:
```bash
npx tsc --noEmit
```

### 5. Docker (optional)

```bash
docker-compose up --build
```

---

## Model Weights

Large model weights are **not committed to git** (excluded by `.gitignore`).

| Model | Location | Size | Download |
|-------|----------|------|----------|
| AttentionChangeNet | `models/satquery_change_v1/attention_best.safetensors` | 48 MB | See `models/satquery_change_v1/provenance.json` |
| FC-EF | `models/satquery_change_v1/fc_ef_best.safetensors` | 7 MB | See provenance |
| SiamUNet-Diff | `models/satquery_change_v1/siamunet_diff_best.safetensors` | 5 MB | See provenance |
| RS-VLM LoRA adapter | `models/satquery_rs_vlm_v1/` | ~4 MB | See `models/satquery_rs_vlm_v1/provenance.json` |
| RS-VQA (873 MB) | `models/satquery_rsvqa_v2/rs_cross_attention_vqa.pt` | 873 MB | See training manifest |

---

## Dataset Provenance

| Dataset | Use | Status |
|---------|-----|--------|
| LEVIR-CD (HuggingFace) | Temporal change detection benchmark | ✅ Used, manifests committed |
| BigEarthNet.txt (HuggingFace) | Multi-label land-cover classification | ⚠️ Experimental (Stage 8.5B/8.6) |

LEVIR-CD split manifests are committed to `data/manifests/`. Parquet files (>50 MB) are git-ignored and reproducible via:
```bash
python scripts/acquire_stage8_6_dataset.py
```

---

## Validation Summary

| Metric | Value | Notes |
|--------|-------|-------|
| AttentionChangeNet F1 (LEVIR-CD held-out) | See `docs/FINAL_HELDOUT_METRICS.json` | Independently benchmarked |
| FC-EF F1 (LEVIR-CD) | See `docs/CHANGE_MODEL_BENCHMARK.json` | |
| BigEarthNet Stage 8.5B | Experimental — see `docs/BIGEARTHNET_STAGE8_5B_RECOVERY_REPORT.md` | ⚠️ Not a final model |
| Prompt 8.6 (RS-VQA) | **Pending** — training in progress | Do not cite |

> Gemini-based review steps within the pipeline are repeated reasoning by the same provider, **not independent scientific validation**.

---

## Known Limitations

- Multi-label BigEarthNet classification (Stage 8.5B) is experimental; accuracy recovery is ongoing
- Prompt 8.6 RS-VQA results are not yet available
- Visual grounding is approximate bounding regions, not pixel-level segmentation
- The frontend Gemini key is held in React memory only (no server-side persistence)
- No authentication or multi-user session management in current prototype
- GeoTIFF upload limited to ≤ 5 MB per image, ≤ 10 MB total per session

---

## Demo Flow

1. Open `http://localhost:3000`
2. Click a **Demo Mission** (Cairo, Florence, or Sundarbans) — no API key needed
3. Or: upload your own GeoTIFF/JPEG/PNG files and add a Gemini API key in Settings
4. Type a natural-language question (e.g., "Has urban area expanded between these dates?")
5. View findings, evidence verdicts, and the execution trace
6. Export as Markdown or download the PDF report

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full system architecture |
| [`docs/FINAL_SYSTEM_VALIDATION.md`](docs/FINAL_SYSTEM_VALIDATION.md) | End-to-end validation results |
| [`docs/FINAL_HELDOUT_METRICS.json`](docs/FINAL_HELDOUT_METRICS.json) | Held-out benchmark numbers |
| [`docs/MODEL_PROVENANCE.md`](docs/MODEL_PROVENANCE.md) | Model training lineage |
| [`docs/BIGEARTHNET_TXT_PROVENANCE.md`](docs/BIGEARTHNET_TXT_PROVENANCE.md) | BigEarthNet dataset sourcing |
| [`docs/DEMO_SCENARIOS.md`](docs/DEMO_SCENARIOS.md) | Demo mission scenarios |
| [`docs/OPTICAL_SAR_VALIDATION.md`](docs/OPTICAL_SAR_VALIDATION.md) | Optical+SAR fusion validation |
| [`docs/DEPLOYMENT_RUNBOOK.md`](docs/DEPLOYMENT_RUNBOOK.md) | Production deployment guide |
| [`public/missions/PROVENANCE.md`](public/missions/PROVENANCE.md) | Demo image attribution |

---

## Attribution & Image Sources

Demo mission images are sourced from NASA Earth Observatory and ESA Copernicus — attribution details in [`public/missions/PROVENANCE.md`](public/missions/PROVENANCE.md).


---

## License

[MIT](LICENSE) — model weights may carry separate licenses from their upstream base models.
