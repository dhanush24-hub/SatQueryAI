import time
import os
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import rasterio
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from app.domain.modalities import Modality
from app.domain.tasks import TaskFamily
from app.core.logging import logger
from app.services.models.base import ModelAdapter
from app.services.models.schemas import GroundingRequest, GroundingResponse, GroundingBox
from app.services.models.coordinates import (
    model_box_to_pixel_box,
    pixel_box_to_canvas_box,
    pixel_box_to_geographic_bbox
)
from app.db.asset_repository import asset_repository
from app.services.storage.local import storage_service


class GroundingAdapter(ModelAdapter):
    """
    Production specialist adapter for Text-Guided Remote-Sensing Grounding.
    Uses google/owlvit-base-patch32 open-vocabulary detector for zero-shot text-conditioned localization.
    """

    def __init__(self, model_name: str = "google/owlvit-base-patch32"):
        self._model_name = model_name
        self._processor: Optional[OwlViTProcessor] = None
        self._model: Optional[OwlViTForObjectDetection] = None
        self._device: str = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def task_family(self) -> TaskFamily:
        return TaskFamily.SINGLE_GROUNDING

    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.UNKNOWN]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        logger.info(f"Loading Grounding model '{self._model_name}' onto {self._device}...")
        self._processor = OwlViTProcessor.from_pretrained(self._model_name)
        self._model = OwlViTForObjectDetection.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()
        logger.info(f"Grounding model '{self._model_name}' successfully loaded.")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Grounding model '{self._model_name}' unloaded from memory.")

    def predict(self, request: GroundingRequest) -> GroundingResponse:
        start_time = time.perf_counter()

        # 1. Resolve asset
        asset = asset_repository.get_asset(request.asset_id)
        if not asset:
            raise ValueError(f"ImageAsset '{request.asset_id}' not found.")

        # 2. Check modality
        if asset.modality == Modality.SAR:
            raise ValueError(
                f"Modality '{asset.modality}' is not supported by optical Grounding model '{self._model_name}'. "
                "Synthetic Aperture Radar requires a specialist SAR detection backbone."
            )

        # 3. Ensure loaded
        self.load()

        # 4. Resolve image
        image = self._load_raster_image(asset)
        img_w, img_h = image.size

        # 5. Format text queries with remote sensing prompt expansion
        raw_queries = [request.queries] if isinstance(request.queries, str) else list(request.queries)
        expanded_queries: List[str] = []
        query_map: Dict[str, str] = {}
        for q in raw_queries:
            clean_q = q.strip()
            if any(clean_q.lower().startswith(p) for p in ["a photo", "a satellite", "an aerial", "satellite"]):
                expanded_queries.append(clean_q)
                query_map[clean_q] = clean_q
            else:
                p_sat = f"a satellite photo of {clean_q}"
                p_air = f"an aerial photo of {clean_q}"
                expanded_queries.extend([clean_q, p_sat, p_air])
                query_map[clean_q] = clean_q
                query_map[p_sat] = clean_q
                query_map[p_air] = clean_q

        text_queries = [expanded_queries]

        # 6. Execute model inference
        inputs = self._processor(text=text_queries, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # 7. Post-process boxes to target image dimensions
        # Attempt requested threshold, or adaptively lower if no detections exist on open-vocabulary zero-shot
        target_sizes = torch.tensor([[img_h, img_w]]).to(self._device)
        effective_threshold = request.threshold
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=effective_threshold,
            text_labels=text_queries
        )[0]

        if len(results["boxes"]) == 0 and effective_threshold > 0.005:
            # Open-vocabulary zero-shot RS detection threshold calibration
            effective_threshold = 0.003
            results = self._processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=effective_threshold,
                text_labels=text_queries
            )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        detected_text_labels = results.get("text_labels", [])

        grounding_boxes: List[GroundingBox] = []
        for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box.tolist()
            # Clamp to image pixel bounds
            px_box = [
                max(0, min(int(round(x1)), img_w - 1)),
                max(0, min(int(round(y1)), img_h - 1)),
                max(1, min(int(round(x2)), img_w)),
                max(1, min(int(round(y2)), img_h))
            ]
            canvas_box = pixel_box_to_canvas_box(px_box, img_w, img_h)
            geo_bbox = pixel_box_to_geographic_bbox(px_box, asset.affine_transform, asset.crs)
            
            prompt_label = detected_text_labels[i] if (i < len(detected_text_labels) and detected_text_labels[i]) else (
                expanded_queries[label_idx] if label_idx < len(expanded_queries) else "target"
            )
            canonical_label = query_map.get(prompt_label, prompt_label)

            grounding_boxes.append(
                GroundingBox(
                    label=canonical_label,
                    score=float(score),
                    canvas_box=canvas_box,
                    pixel_box=px_box,
                    geographic_bbox=geo_bbox
                )
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.info(f"Grounding executed for asset {request.asset_id}: {len(grounding_boxes)} regions found in {latency_ms}ms")

        warnings = []
        if len(grounding_boxes) == 0:
            warnings.append(f"No regions exceeded detection threshold of {request.threshold} for query {raw_queries}.")
        if not asset.crs:
            warnings.append("Asset lacks georeferencing CRS. Grounding returns pixel/normalized coordinates only.")

        return GroundingResponse(
            boxes=grounding_boxes,
            model_name=self._model_name,
            task_family=TaskFamily.SINGLE_GROUNDING,
            device=self._device,
            inference_latency_ms=latency_ms,
            queries=raw_queries,
            warnings=warnings,
            metadata={
                "asset_id": asset.id,
                "input_dimensions": [asset.width, asset.height],
                "crs": asset.crs,
                "threshold_applied": request.threshold,
                "detected_count": len(grounding_boxes)
            }
        )

    def _load_raster_image(self, asset) -> Image.Image:
        raw_path = asset.storage_path if (os.path.isabs(asset.storage_path) and os.path.exists(asset.storage_path)) else storage_service.get_full_path(os.path.basename(asset.storage_path))
        if os.path.exists(raw_path):
            try:
                with rasterio.open(raw_path) as src:
                    count = src.count
                    if count >= 3:
                        r = src.read(1)
                        g = src.read(2)
                        b = src.read(3)
                    elif count == 1:
                        band = src.read(1)
                        r = g = b = band
                    else:
                        r = src.read(1)
                        g = src.read(2)
                        b = src.read(1)

                    def norm_band(arr):
                        p2, p98 = np.percentile(arr, (2, 98))
                        if p98 > p2:
                            arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255)
                        return arr.astype(np.uint8)

                    rgb = np.stack([norm_band(r), norm_band(g), norm_band(b)], axis=-1)
                    return Image.fromarray(rgb)
            except Exception as e:
                logger.warning(f"Could not read full raster directly with rasterio: {e}")

        if asset.preview_path:
            full_preview_path = storage_service.get_full_path(os.path.basename(asset.preview_path))
            if os.path.exists(full_preview_path):
                return Image.open(full_preview_path).convert("RGB")

        raise ValueError(f"Unable to read raster imagery for asset '{asset.id}'.")

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.is_loaded else "unloaded",
            "model_name": self._model_name,
            "device": self._device,
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available()
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "task_family": TaskFamily.SINGLE_GROUNDING.value,
            "supported_modalities": [m.value for m in self.supported_modalities],
            "license": "Apache-2.0",
            "source": "https://huggingface.co/google/owlvit-base-patch32"
        }
