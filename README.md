# SatQuery AI — Alpha Logic / SIH 2026 / PS 26167

SatQuery turns imagery plus a natural-language question into an automatically composed, evidence-reviewed analysis. Users never choose Observe, Compare or Fuse. The existing private Site and landing/workspace are preserved.

## Current prototype architecture

Images → input assessment → SatQuery planner → deterministic capability composition → replaceable Gemini capability adapters → grounding review → uncertainty review → evidence gate → synthesis from gated findings.

`lib/domain.ts` defines workflow capabilities/steps, relationship assessments, first-class findings, evidence, verdicts and execution events. `lib/workflow.ts` composes a bounded, dependency-ordered executable plan from interpreted intent and image relationships. The planner does not produce final findings.

`lib/registry.ts` exposes capability/provider interfaces and a registry. Dedicated prompt modules cover scene, temporal, optical, SAR display, cross-sensor, grounding, uncertainty, validation and synthesis. Gemini is a temporary provider behind this interface, not the product architecture. Each selected capability performs its own actual request. The executor dispatches steps, validates responses, enforces the evidence gate and emits NDJSON status events as the requests complete. A provider error stops the pipeline; it never substitutes demo findings.

The pipeline can compose multiple capabilities, sequential comparisons across three or more dated observations, and independent optical/radar extraction before cross-sensor comparison. A description of unrelated images needs no relationship or acquisition metadata. Temporal questions need order and scene suitability. Sensor-specific questions need identified inputs. Provided metadata is never labeled independently verified; filenames and visual sensor guesses remain inferences.

## Evidence and uncertainty

Findings separate observations, likely interpretations and uncertainty. Every evidence ID must exist, be linked to its finding, and refer to an available image. Percentage regions must stay in bounds. Grounding remains approximate, not segmentation.

Every candidate gets a gate verdict: supported, partially supported, insufficient or conflicting. Missing visual grounding and missing temporal counterpart evidence prevent support. Low support is downgraded; explicit sensor disagreement forces a conflicting verdict. Failed evidence is retained as uncertain with reasons and requests for better observations, not silently promoted. Final synthesis follows the gate. No-support results are forced to abstain. Gemini reviews are repeated reasoning by the same provider, **not independent scientific validation**.

## Mission interaction

- Upload 1–6 PNG/JPEG/WebP files, at most 5 MB each / 10 MB total.
- Browser decoding and dimensions checks; server signatures, limits and duplicate validation.
- Optional image context accepts dates, sensor and location. No mode selection.
- Missing material context prompts a clarification; the original question and images remain in the mission.
- Follow-ups pass prior relationships, findings, evidence and verdicts. Questions about alternatives run a review-only pipeline over existing candidates. High-support filtering uses existing finding/verdict data locally without a new model call.
- Select a finding to see associated regions side by side. Switching images retains the corresponding selected finding reference.
- The workflow panel explains intent, source provenance, relationships and actual step statuses. The trace names inputs, purpose, provider, timestamp, status and output summary.
- Markdown export includes findings, verdicts, alternatives, evidence and the execution trace.

## Demonstration missions

Demos run curated fixture providers through the **same executor and deterministic evidence gate**. They are explicitly labeled curated, not live AI. They never fake specialist model execution.

1. **Cairo: a changing city** — Landsat optical acquisitions, 1984-07-02 and 2019-09-05. Apparent built-up expansion is paired with an uncertain vegetation-appearance candidate. A seasonal follow-up retains the existing evidence.
2. **Florence: incomplete agreement** — a 2018-09-19 Landsat false-color optical view and a source-published 2018-09-14 Sentinel-1-derived ARIA flood proxy. Different dates and footprints prevent local cross-sensor confirmation. The proxy is not raw SAR and is not produced by SatQuery.
3. **Sundarbans: read the landscape** — archival Landsat 7 mosaic, November 1999 / November 2000. A single display mosaic, not a temporal pair.

Image attribution and source details are in `public/missions/PROVENANCE.md`. Cairo: NASA Earth Observatory / Lauren Dauphin, Landsat/USGS. Florence optical: Joshua Stevens, Landsat/USGS. Radar-derived map: Joshua Stevens and Lauren Dauphin; modified Copernicus Sentinel data (2018), ESA and NASA-JPL/Caltech ARIA. Sundarbans: Jesse Allen, University of Maryland Global Land Cover Facility.

## Gemini and privacy

Use workspace settings for an API key and model name. The key remains in React memory, clears on refresh, and is sent through same-origin server endpoints to Google. This application does not durably store keys, imagery or conversations. Google processing policies still apply. Do not commit credentials.

Live multi-stage requests take longer and consume more provider quota than a single answer call. Each stage has a timeout and cancellation; errors remain visible. Live success has not been verified with an actual user key. Provider tests use mocks and deterministic fixture adapters.

## Target SatQuery architecture

Satellite data → geo-harmonization → SatQuery agent → specialist remote-sensing model registry → scene / temporal / optical-SAR analysis → evidence gate → re-plan if necessary → verifiable geospatial intelligence.

Future providers can implement scene VQA, change detection, real radar analysis, grounding/segmentation or independent validation without replacing workspace findings/evidence rendering. No fake future tools are implemented.

Out of scope in this prototype: GeoTIFF/raw SAR parsing, GDAL, CRS transforms, real co-registration, segmentation, NDVI, calibrated confidence, quantitative area measurement, authentication, billing or generic GIS/SaaS features.

## Development and validation

- `npm install`
- `npm run dev`
- `npm run build`
- `npx tsc --noEmit`
- `node --test tests/analysis.test.mjs`

Tests cover all 15 requested scenarios plus execution ordering, complete gate coverage, demo pipelines, source/signature guards, temporal counterpart grounding, conflicting evidence, provider failure and local follow-up filtering.

No browser interaction/visual QA or live Gemini success is claimed. The optional feature-detected WebMCP tool stages a question without submitting it; a supported WebMCP validation context was unavailable.

Reuse `.openai/hosting.json` and existing Sites project `appgprj_6a9d5e0b37b08191b3cad577a116a425`. Keep the parent `sources/` folder and reference PDFs read-only. Never persist source credential tokens.
