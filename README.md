# SatQuery AI

Alpha Logic SIH 2026 prototype, PS 26167. Landing page and question-first workspace with automatic workflow composition.

Run `npm install`, then `npm run dev`. Production build: `npm run build`. Checks: `npx tsc --noEmit` and `node --test tests/analysis.test.mjs` (input limits/signatures/duplicates, automatic planning, clarification gates, multi-image evidence and provider errors).

## Demonstration

Choose Explore a sample mission, submit the suggested question, select an evidence card, inspect the trail and limitations, and export the report. The sample is curated; it does not call Gemini. NASA Landsat 7 Sundarbans imagery is an archival November 1999 / November 2000 mosaic, not a time-series pair. Attribution: NASA Earth Observatory / Jesse Allen, using University of Maryland Global Land Cover Facility data. Source: https://science.nasa.gov/earth/earth-observatory/sundarbans-bangladesh-7028/

Use connection settings to provide a Gemini key and model identifier, then enable live analysis. The key stays only in React memory and is sent over HTTPS to the same-origin analysis endpoint, which forwards it to Google. No key, images, or conversation is stored by this application. A refresh clears the session. Google provider processing policies still apply. Do not commit credentials.

Upload 1–6 images (5 MB each, 10 MB total). There is no mode selector. File decoding and duplicate checks run locally; the server also checks file signatures and size limits. A separate Gemini input-assessment request selects and composes scene, temporal, and sensor-fusion reasoning. A clarification gate stops analysis when required context is absent. The user can answer naturally or attach missing imagery. A second Gemini request follows the selected plan. Timestamped execution events record actual request stages. PNG, JPEG, and WebP display exports up to 5 MB each are supported. Acquisition labels are user supplied and unverified. The planner assesses sensor identity, dates, and region comparability without assuming upload order or treating grayscale as SAR. User labels and filename clues remain unverified. The model provides qualitative visual interpretation and approximate evidence boxes. No GDAL, GeoTIFF parsing, alignment, calibrated confidence, segmentation, NDVI, area measurement, or specialist remote-sensing model execution is implemented. The architecture leaves those as a future processing backend rather than falsely claiming them in this prototype.

A feature-detected WebMCP tool stages a question without submitting it. No supported WebMCP validation context was available during development; browser contract validation remains unverified. Live provider success needs the user's valid Gemini key; automated adapter tests use mocked provider responses.
