# SatQuery AI: Production Deployment Runbook

## 1. System Architecture

SatQuery AI is architected as an offline-capable, decoupled geospatial intelligence system:

```
┌─────────────────────────────────────────────────────────────┐
│               Frontend: Next.js / React 19                 │
│         (Analysis Workspace, Pan/Zoom, Swipe Viewer)        │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / REST API (:8000)
┌──────────────────────────────▼──────────────────────────────┐
│                Backend: FastAPI / Python 3.12                │
│    Orchestration Controller → Model Adapters → Geospatial    │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
┌──────────────▼──────────────┐┌──────────────▼───────────────┐
│     SQLite DB (satquery.db) ││ Local Raster Storage         │
│  - Asset Provenance         ││ - storage/rasters/           │
│  - Analysis History         ││ - storage/masks/             │
│  - Execution Traces         ││ - storage/previews/          │
└─────────────────────────────┘└──────────────────────────────┘
```

---

## 2. Quickstart: Local Development

### Prerequisites
- Python 3.12+
- Node.js 20+
- GDAL / GEOS libraries (`brew install gdal` on macOS, `apt-get install libgdal-dev` on Linux)

### Backend Setup
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Launch backend API server
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be accessible at: `http://localhost:8000/docs`

### Frontend Setup
```bash
# 1. Install Node dependencies
npm install

# 2. Launch Next.js development server
npm run dev
```
Web Workspace will be accessible at: `http://localhost:3000`

---

## 3. Docker Production Deployment

To run the fully containerized multi-tier stack:

```bash
# Build and launch all services in detached mode
docker-compose up -d --build

# Verify container health
docker-compose ps

# Tail backend logs
docker-compose logs -f backend
```

---

## 4. Hardware Accelerator Auto-Detection & Fallback

The backend dynamically detects and leverages available hardware without code modifications:

1. **NVIDIA CUDA:**
   - Automatically selected if `torch.cuda.is_available() == True`.
   - Uses `bfloat16` / `float16` for high-throughput batch evaluation.
2. **Apple Silicon (MPS):**
   - Automatically selected if `torch.backends.mps.is_available() == True`.
   - Uses unified system memory with MPS-accelerated convolution and matrix multiplication.
3. **CPU Fallback:**
   - If no dedicated accelerator is detected, all models load seamlessly onto CPU (`float32`).
   - PyTorch OpenMP multi-threading is utilized automatically.

To force CPU execution regardless of hardware, set:
```bash
export TORCH_DEVICE=cpu
```

---

## 5. Offline Operation & Model Caching

SatQuery AI is designed to operate in air-gapped / disconnected disaster relief environments:

1. **Pre-Cached Checkpoints:**
   - AttentionChangeNet weights: `models/satquery_change_v1/attention_best.safetensors` (local file, zero internet dependency).
   - Hugging Face cache: set `HF_HOME=/app/models/cache` to ensure pre-downloaded transformer weights are read exclusively from disk.
2. **No External Network Calls:**
   - Map tiling, coordinate projection, and analysis execution run completely locally without Google Maps or Mapbox API keys.

---

## 6. Healthchecks & Diagnostics

### API Health Endpoint
```bash
curl -f http://localhost:8000/api/health
```
Expected Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "mps",
  "models_ready": true
}
```

### Full Automated Regression Check
```bash
# Run 92 backend tests
PYTHONPATH=backend pytest backend/tests -v

# Run 27 frontend tests
node --test tests/analysis.test.mjs

# Run 10 mandatory real-world E2E scenarios
PYTHONPATH=backend python scripts/verify_prompt8_e2e.py
```
