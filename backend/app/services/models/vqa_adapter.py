import time
import os
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import rasterio
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

from app.domain.modalities import Modality
from app.domain.tasks import TaskFamily
from app.core.logging import logger
from app.services.models.base import ModelAdapter
from app.services.models.schemas import VqaRequest, VqaResponse
from app.db.asset_repository import asset_repository
from app.services.storage.local import storage_service


class VqaAdapter(ModelAdapter):
    """
    Production specialist adapter for Single-Image Remote-Sensing VQA.
    Uses Salesforce/blip-vqa-base with lazy loading and explicit device placement.
    """

    def __init__(self, model_name: str = "Salesforce/blip-vqa-base"):
        self._model_name = model_name
        self._processor: Optional[BlipProcessor] = None
        self._model: Optional[BlipForQuestionAnswering] = None
        self._device: str = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def task_family(self) -> TaskFamily:
        return TaskFamily.SINGLE_VQA

    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.UNKNOWN]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        logger.info(f"Loading VQA model '{self._model_name}' onto {self._device}...")
        self._processor = BlipProcessor.from_pretrained(self._model_name)
        self._model = BlipForQuestionAnswering.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()
        logger.info(f"VQA model '{self._model_name}' successfully loaded.")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"VQA model '{self._model_name}' unloaded from memory.")

    def predict(self, request: VqaRequest) -> VqaResponse:
        start_time = time.perf_counter()

        # 1. Resolve asset
        asset = asset_repository.get_asset(request.asset_id)
        if not asset:
            raise ValueError(f"ImageAsset '{request.asset_id}' not found.")

        # 2. Check modality compatibility
        if asset.modality == Modality.SAR:
            raise ValueError(
                f"Modality '{asset.modality}' is not supported by optical VQA model '{self._model_name}'. "
                "Synthetic Aperture Radar requires a specialist SAR vision-language backbone."
            )

        # 3. Ensure model loaded
        self.load()

        # 4. Resolve and preprocess actual raster
        image = self._load_raster_image(asset)

        # 5. Run actual inference
        inputs = self._processor(image, request.question, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=request.max_tokens
            )
        answer = self._processor.decode(outputs[0], skip_special_tokens=True).strip()

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.info(f"VQA executed for asset {request.asset_id} in {latency_ms}ms: '{answer}'")

        warnings = []
        if asset.modality == Modality.UNKNOWN:
            warnings.append("Modality is UNKNOWN; evaluated using standard 3-band RGB optical interpretation.")
        if not asset.crs:
            warnings.append("Asset lacks georeferencing CRS. Responses are purely qualitative visual observations.")

        return VqaResponse(
            answer=answer,
            model_name=self._model_name,
            task_family=TaskFamily.SINGLE_VQA,
            device=self._device,
            inference_latency_ms=latency_ms,
            confidence=None,  # Uncalibrated softmax outputs are not returned as calibrated confidence
            warnings=warnings,
            metadata={
                "asset_id": asset.id,
                "input_dimensions": [asset.width, asset.height],
                "crs": asset.crs,
                "resolution_gsd": asset.resolution,
                "sensor": asset.sensor
            }
        )

    def _load_raster_image(self, asset) -> Image.Image:
        """Load and normalize original raster or preview into a PIL RGB Image."""
        # Check storage file
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

                    # Normalize to 8-bit
                    def norm_band(arr):
                        p2, p98 = np.percentile(arr, (2, 98))
                        if p98 > p2:
                            arr = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255)
                        return arr.astype(np.uint8)

                    rgb = np.stack([norm_band(r), norm_band(g), norm_band(b)], axis=-1)
                    return Image.fromarray(rgb)
            except Exception as e:
                logger.warning(f"Could not read full raster directly with rasterio: {e}")

        # Fallback to preview path if available
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
            "task_family": TaskFamily.SINGLE_VQA.value,
            "supported_modalities": [m.value for m in self.supported_modalities],
            "license": "BSD-3-Clause",
            "source": "https://huggingface.co/Salesforce/blip-vqa-base"
        }
