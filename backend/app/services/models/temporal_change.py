import io
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import rasterio
from PIL import Image
import torch

from app.domain.modalities import Modality
from app.domain.tasks import TaskFamily
from app.schemas.image_asset import ImageAsset
from app.db.asset_repository import asset_repository
from app.services.storage.local import storage_service
from app.services.geospatial.temporal_validation import validate_temporal_pair, TemporalValidationError
from app.services.geospatial.registration import verify_registration_quality, RegistrationAssessment
from app.services.geospatial.alignment import align_and_reproject_raster
from app.services.geospatial.area_calculation import calculate_changed_area
from app.services.models.base import ModelAdapter
from app.services.models.schemas import (
    TemporalChangeRequest,
    TemporalChangeResponse,
    TemporalChangeCluster
)
from app.services.models.siamunet_diff import SiamUnet_diff, load_siamunet_diff_model
from app.services.models.attention_change import AttentionChangeNet
from app.services.geospatial.domain_gate import assess_input_domain, DomainSuitability
from app.services.models.coordinates import pixel_box_to_canvas_box, pixel_box_to_geographic_bbox
from app.core.logging import logger
from safetensors import safe_open

try:
    from scipy.ndimage import label, find_objects
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TemporalChangeAdapter(ModelAdapter):
    """
    Specialist bi-temporal change detection adapter.
    Executes real inference using fine-tuned AttentionChangeNet (winning candidate),
    fine-tuned SiamUnet_diff, or an analytical Change Vector Analysis (CVA) baseline.
    Protected by domain suitability gate and calibrated thresholds.
    """

    DEFAULT_CHECKPOINT = "models/satquery_change_v1/attention_best.safetensors"
    FALLBACK_REPO_ID = "HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff"
    CALIBRATED_THRESHOLD = 0.50

    def __init__(self, device: Optional[str] = None):
        self._device_str = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: Optional[torch.nn.Module] = None
        self._active_model_name: str = "AttentionChangeNet"
        self._is_loaded = False

    @property
    def model_name(self) -> str:
        return f"satquery/{self._active_model_name.lower()}-v1"

    @property
    def task_family(self) -> TaskFamily:
        return TaskFamily.TEMPORAL_CHANGE

    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.OPTICAL, Modality.MULTISPECTRAL]

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None

    def load(self) -> None:
        if self._model is None:
            if os.path.exists(self.DEFAULT_CHECKPOINT):
                logger.info(f"Loading winning fine-tuned AttentionChangeNet from {self.DEFAULT_CHECKPOINT} on {self._device_str}...")
                model = AttentionChangeNet(num_classes=2, pretrained=False)
                state_dict = {}
                with safe_open(self.DEFAULT_CHECKPOINT, framework="pt") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
                model.load_state_dict(state_dict)
                self._model = model.to(self._device_str)
                self._model.eval()
                self._active_model_name = "AttentionChangeNet"
                self._is_loaded = True
            else:
                logger.warning(f"Local checkpoint {self.DEFAULT_CHECKPOINT} not found, falling back to SiamUnet_diff...")
                self._model = load_siamunet_diff_model(repo_id=self.FALLBACK_REPO_ID, device=self._device_str)
                self._active_model_name = "SiamUnet_diff"
                self._is_loaded = True

    def unload(self) -> None:
        if self._model is not None:
            logger.info(f"Unloading change model from memory...")
            del self._model
            self._model = None
            self._is_loaded = False
            if self._device_str == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif self._device_str == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.is_loaded else "unloaded",
            "model_name": self.model_name,
            "device": self._device_str
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture": self._active_model_name,
            "training_dataset": "LEVIR-CD256 (ericyu/LEVIRCD_Cropped256)",
            "license": "Apache-2.0 / MIT",
            "task": self.task_family.value,
            "calibrated_threshold": self.CALIBRATED_THRESHOLD,
            "supported_modalities": [m.value for m in self.supported_modalities]
        }

    def predict(self, request: TemporalChangeRequest) -> TemporalChangeResponse:
        t0 = time.time()
        warnings: List[str] = []

        # 1. Resolve and validate assets
        a1 = asset_repository.get_asset(request.t1_asset_id)
        a2 = asset_repository.get_asset(request.t2_asset_id)
        if not a1 or not a2:
            raise ValueError(f"One or both assets not found: '{request.t1_asset_id}', '{request.t2_asset_id}'.")

        # Temporal pair validation
        user_order = request.parameters.get("user_order")
        t1_asset, t2_asset, temp_meta = validate_temporal_pair([a1, a2], user_order=user_order)

        # 2. Read raw raster imagery
        paths1 = asset_repository.get_internal_paths(t1_asset.id)
        t1_path = paths1["storage_path"] if (paths1 and paths1.get("storage_path") and os.path.exists(paths1["storage_path"])) else None
        if not t1_path:
            t1_path = t1_asset.storage_path if (os.path.isabs(t1_asset.storage_path) and os.path.exists(t1_asset.storage_path)) else storage_service.get_full_path(os.path.basename(t1_asset.storage_path))

        paths2 = asset_repository.get_internal_paths(t2_asset.id)
        t2_path = paths2["storage_path"] if (paths2 and paths2.get("storage_path") and os.path.exists(paths2["storage_path"])) else None
        if not t2_path:
            t2_path = t2_asset.storage_path if (os.path.isabs(t2_asset.storage_path) and os.path.exists(t2_asset.storage_path)) else storage_service.get_full_path(os.path.basename(t2_asset.storage_path))

        # Check if alignment / reprojection is needed
        if temp_meta.get("needs_reprojection") or not temp_meta.get("is_grid_aligned"):
            logger.info(f"Re-projecting/aligning T2 asset '{t2_asset.id}' to T1 grid '{t1_asset.id}'...")
            try:
                aligned_t2 = align_and_reproject_raster(source_asset_id=t2_asset.id, reference_asset_id=t1_asset.id)
                t2_paths = asset_repository.get_internal_paths(aligned_t2.id)
                t2_aligned_path = t2_paths["storage_path"] if t2_paths else None
                if not t2_aligned_path or not os.path.exists(t2_aligned_path):
                    t2_aligned_path = storage_service.get_full_path(os.path.basename(aligned_t2.storage_path))
                warnings.append("Images reprojected and grid-aligned to common spatial extent.")
            except Exception as e:
                logger.warning(f"Alignment step skipped due to: {e}")
                t2_aligned_path = t2_path
        else:
            t2_aligned_path = t2_path

        # Load raster arrays
        with rasterio.open(t1_path) as src1:
            arr1 = src1.read().astype(np.float32)  # (bands, H, W)
            profile1 = src1.profile
            affine1 = list(src1.transform)
            crs1 = str(src1.crs)
            nodata1 = src1.nodata

        with rasterio.open(t2_aligned_path) as src2:
            arr2 = src2.read().astype(np.float32)
            nodata2 = src2.nodata

        # Align dimensions if slight 1-pixel boundary discrepancy
        min_h = min(arr1.shape[1], arr2.shape[1])
        min_w = min(arr1.shape[2], arr2.shape[2])
        arr1 = arr1[:, :min_h, :min_w]
        arr2 = arr2[:, :min_h, :min_w]

        # 3. Registration Verification (Empirical subpixel translation & correlation)
        reg_assessment = verify_registration_quality(arr1, arr2, nodata_val=nodata1)
        if reg_assessment.warnings:
            warnings.extend(reg_assessment.warnings)

        # 3.5 Radiometric valid-data mask and Input Domain Suitability Assessment
        valid_mask = np.ones((min_h, min_w), dtype=bool)
        if nodata1 is not None:
            valid_mask &= (arr1[0] != nodata1)
        if nodata2 is not None:
            valid_mask &= (arr2[0] != nodata2)

        nodata_ratio = 1.0 - (float(np.sum(valid_mask)) / max(1, min_h * min_w))
        domain_check = assess_input_domain(
            modality=str(t1_asset.modality.value if hasattr(t1_asset.modality, "value") else t1_asset.modality),
            num_bands=arr1.shape[0],
            dimensions=(min_h, min_w),
            resolution_m=t1_asset.resolution,
            registration_status=reg_assessment.quality_status,
            cloud_or_nodata_ratio=max(0.0, nodata_ratio),
            task="temporal_change"
        )
        if domain_check.warnings:
            warnings.extend(domain_check.warnings)
        if not domain_check.is_suitable:
            warnings.extend(domain_check.reasons)
            warnings.append("Input out of domain: " + "; ".join(domain_check.reasons))
            return TemporalChangeResponse(
                task_family=TaskFamily.TEMPORAL_CHANGE,
                model_name=self.model_name,
                device=self._device_str,
                inference_latency_ms=round((time.time() - t0) * 1000.0, 1),
                change_clusters=[],
                total_changed_pixels=0,
                total_valid_pixels=min_h * min_w,
                change_ratio_pct=0.0,
                has_authentic_geo=False,
                change_verdict="INSUFFICIENT_EVIDENCE",
                mask_preview_url=None,
                registration_assessment=reg_assessment.model_dump() if reg_assessment else None,
                confidence=None,
                warnings=warnings,
                metadata={"domain_suitability": domain_check.model_dump()}
            )

        # 4. Radiometric and valid-data normalization
        # Standardize to 3-band RGB in [0, 1] using reference-based percentile scaling
        def prepare_rgb(arr: np.ndarray, ref_p2=None, ref_p98=None) -> Tuple[np.ndarray, List[float], List[float]]:
            if arr.shape[0] == 1:
                rgb = np.repeat(arr, 3, axis=0)
            elif arr.shape[0] >= 3:
                rgb = arr[:3]
            else:
                rgb = np.pad(arr, ((0, 3 - arr.shape[0]), (0, 0), (0, 0)), mode="edge")
            
            stretched = np.zeros_like(rgb, dtype=np.float32)
            calc_percentiles = (ref_p2 is None or ref_p98 is None)
            p2s, p98s = [], []
            for b in range(3):
                band = rgb[b]
                v_valid = band[valid_mask] if np.any(valid_mask) else band
                if calc_percentiles:
                    p2, p98 = float(np.percentile(v_valid, 2)), float(np.percentile(v_valid, 98))
                else:
                    p2, p98 = ref_p2[b], ref_p98[b]
                p2s.append(p2)
                p98s.append(p98)
                if p98 > p2:
                    stretched[b] = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
                else:
                    max_val = np.max(band) if np.max(band) > 0 else 1.0
                    stretched[b] = np.clip(band / max_val, 0.0, 1.0)
            return stretched, p2s, p98s

        rgb1, p2_ref, p98_ref = prepare_rgb(arr1)
        rgb2, _, _ = prepare_rgb(arr2, ref_p2=p2_ref, ref_p98=p98_ref)

        # 5. Execute Change Detection
        method = request.parameters.get("method") or request.method or "attention"
        default_thresh = self.CALIBRATED_THRESHOLD if method in ["attention", "siamunet_diff"] else 0.50
        threshold = float(request.parameters.get("threshold") or request.threshold or default_thresh)

        # Identical image check: if identical arrays, bypass model to guarantee exact no-change baseline
        is_identical = np.array_equal(arr1, arr2) or np.allclose(arr1, arr2, atol=1e-4)
        if is_identical:
            logger.info("Identical inputs detected: perfect no-change baseline.")
            change_prob_map = np.zeros((min_h, min_w), dtype=np.float32)
            binary_mask = np.zeros((min_h, min_w), dtype=bool)
        elif method == "analytical_cva":
            # Change Vector Analysis baseline
            diff_vec = rgb2 - rgb1
            cva_mag = np.linalg.norm(diff_vec, axis=0)  # Euclidean magnitude
            # Adaptive threshold
            cva_valid = cva_mag[valid_mask] if np.any(valid_mask) else cva_mag
            mu = float(np.mean(cva_valid))
            sigma = float(np.std(cva_valid))
            cva_thresh = mu + 1.8 * sigma
            change_prob_map = np.clip(cva_mag / (cva_thresh * 1.5 + 1e-6), 0.0, 1.0)
            binary_mask = (cva_mag >= cva_thresh) & valid_mask
        else:
            # Neural network inference via calibrated winning model
            self.load()
            binary_mask, change_prob_map = self._run_tiled_inference(rgb1, rgb2, threshold=threshold)
            binary_mask &= valid_mask

        total_valid_pixels = int(np.sum(valid_mask))
        total_changed_pixels = int(np.sum(binary_mask))
        change_ratio_pct = round((total_changed_pixels / total_valid_pixels) * 100.0, 4) if total_valid_pixels > 0 else 0.0

        # 6. Physical Area Calculation (Zero-Fabrication Policy)
        has_authentic_geo = False
        if crs1 and str(crs1).strip() not in ["", "None", "None/None"]:
            if affine1 is not None and len(affine1) >= 6:
                # Authentic if not identity transform [1, 0, 0, 0, 1, 0]
                if not (affine1[0] == 1.0 and affine1[4] == 1.0 and affine1[1] == 0.0 and affine1[3] == 0.0):
                    has_authentic_geo = True

        center_lat = (t1_asset.geographic_bbox[1] + t1_asset.geographic_bbox[3]) / 2.0 if (t1_asset.geographic_bbox and has_authentic_geo) else 0.0
        
        if has_authentic_geo:
            area_stats = calculate_changed_area(
                num_changed_pixels=total_changed_pixels,
                crs_str=crs1,
                resolution=t1_asset.resolution,
                center_lat=center_lat,
                affine_transform=affine1
            )
            changed_area_m2 = area_stats["sq_meters"]
            changed_area_ha = area_stats["hectares"]
            changed_area_sq_km = area_stats["sq_km"]
        else:
            changed_area_m2 = None
            changed_area_ha = None
            changed_area_sq_km = None
            warnings.append("Area calculation omitted: Input rasters lack authentic geospatial reference metadata (zero-fabrication policy).")

        # 7. Extract Spatial Change Clusters
        clusters = self._extract_clusters(
            binary_mask=binary_mask,
            prob_map=change_prob_map,
            img_dims=(min_w, min_h),
            affine=affine1 if has_authentic_geo else None,
            crs_str=crs1 if has_authentic_geo else None,
            center_lat=center_lat,
            resolution=t1_asset.resolution if has_authentic_geo else None,
            has_authentic_geo=has_authentic_geo
        )

        # 8. Determine Scientific Change Verdict
        if not reg_assessment.is_sufficient_for_pixel_localization:
            change_verdict = "INSUFFICIENT_EVIDENCE"
            warnings.append("Registration uncertainty exceeds tolerance. Spatial change findings cannot be validated.")
        elif total_changed_pixels == 0 or change_ratio_pct < 0.05:
            change_verdict = "NO_CHANGE_DETECTED"
        elif change_ratio_pct >= 0.50 and reg_assessment.quality_status in ["HIGH", "ACCEPTABLE"]:
            change_verdict = "OBSERVED_CHANGE"
        else:
            change_verdict = "POSSIBLE_CHANGE"
            warnings.append("Change magnitude is marginal or partially impacted by radiometric / registration variation.")

        # 9. Save visual change mask preview PNG
        preview_url = self._save_change_mask_preview(binary_mask, t1_asset.id, t2_asset.id)

        duration_ms = (time.time() - t0) * 1000.0

        return TemporalChangeResponse(
            task_family=TaskFamily.TEMPORAL_CHANGE,
            model_name="analytical_cva" if method == "analytical_cva" else self.model_name,
            device="cpu" if method == "analytical_cva" else self._device_str,
            inference_latency_ms=round(duration_ms, 1),
            change_clusters=clusters,
            total_changed_pixels=total_changed_pixels,
            total_valid_pixels=total_valid_pixels,
            change_ratio_pct=change_ratio_pct,
            changed_area_m2=changed_area_m2,
            changed_area_ha=changed_area_ha,
            changed_area_sq_km=changed_area_sq_km,
            has_authentic_geo=has_authentic_geo,
            change_verdict=change_verdict,
            mask_preview_url=preview_url,
            registration_assessment=reg_assessment.model_dump(),
            confidence=None,
            warnings=warnings,
            metadata={
                "temporal_pair": temp_meta,
                "method": method,
                "threshold": threshold,
                "center_lat": center_lat,
                "resolution": t1_asset.resolution,
                "has_authentic_geo": has_authentic_geo
            }
        )

    def _run_tiled_inference(
        self,
        rgb1: np.ndarray,
        rgb2: np.ndarray,
        threshold: float = 0.50,
        tile_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs tiled/windowed inference for arbitrary image dimensions with seam smoothing.
        """
        _, h, w = rgb1.shape
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        # Compute tiles
        stride = tile_size - 32  # 32px overlap
        y_starts = list(range(0, max(1, h - tile_size + 1), stride))
        if not y_starts or y_starts[-1] + tile_size < h:
            y_starts.append(max(0, h - tile_size))

        x_starts = list(range(0, max(1, w - tile_size + 1), stride))
        if not x_starts or x_starts[-1] + tile_size < w:
            x_starts.append(max(0, w - tile_size))

        self._model.eval()
        with torch.no_grad():
            for ys in y_starts:
                for xs in x_starts:
                    ye = min(h, ys + tile_size)
                    xe = min(w, xs + tile_size)

                    patch1 = rgb1[:, ys:ye, xs:xe]
                    patch2 = rgb2[:, ys:ye, xs:xe]

                    ph, pw = patch1.shape[1], patch1.shape[2]
                    # Pad to tile_size if edge patch
                    if ph < tile_size or pw < tile_size:
                        pad_h = tile_size - ph
                        pad_w = tile_size - pw
                        patch1 = np.pad(patch1, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                        patch2 = np.pad(patch2, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

                    # ImageNet normalization standard for UCD change models
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    p1_norm = (patch1 - mean) / std
                    p2_norm = (patch2 - mean) / std

                    t1_tensor = torch.from_numpy(p1_norm).unsqueeze(0).to(self._device_str)
                    t2_tensor = torch.from_numpy(p2_norm).unsqueeze(0).to(self._device_str)

                    # Forward pass
                    out = self._model(t1_tensor, t2_tensor)  # (1, 2, 256, 256)
                    # Exponentiate log-softmax to obtain probability of change (class 1)
                    change_prob = torch.exp(out[:, 1]).squeeze(0).cpu().numpy()[:ph, :pw]

                    prob_map[ys:ye, xs:xe] += change_prob
                    count_map[ys:ye, xs:xe] += 1.0

        count_map = np.maximum(count_map, 1.0)
        avg_prob_map = prob_map / count_map
        binary_mask = (avg_prob_map >= threshold)

        return binary_mask, avg_prob_map

    def _extract_clusters(
        self,
        binary_mask: np.ndarray,
        prob_map: np.ndarray,
        img_dims: Tuple[int, int],
        affine: Optional[List[float]],
        crs_str: Optional[str],
        center_lat: Optional[float],
        resolution: Optional[float],
        has_authentic_geo: bool = False,
        min_cluster_pixels: int = 15
    ) -> List[TemporalChangeCluster]:
        """
        Clusters contiguous changed regions into spatial findings with true physical area.
        Strictly prevents geometry or area fabrication if authentic georeferencing is missing.
        """
        clusters: List[TemporalChangeCluster] = []
        w, h = img_dims

        if not HAS_SCIPY or not np.any(binary_mask):
            return clusters

        structure = np.ones((3, 3), dtype=int)
        labeled_arr, num_features = label(binary_mask, structure=structure)
        objects = find_objects(labeled_arr)

        cluster_idx = 1
        for i, slice_tuple in enumerate(objects):
            if slice_tuple is None:
                continue
            y_slice, x_slice = slice_tuple
            cluster_mask = (labeled_arr[y_slice, x_slice] == (i + 1))
            cluster_pixel_count = int(np.sum(cluster_mask))

            if cluster_pixel_count < min_cluster_pixels:
                continue

            ymin, ymax = y_slice.start, y_slice.stop
            xmin, xmax = x_slice.start, x_slice.stop

            px_box = [xmin, ymin, xmax, ymax]
            canvas_box = pixel_box_to_canvas_box(px_box, w, h)

            geo_bbox = None
            centroid_geo = None
            geometry = None
            area_m2 = None
            area_ha = None

            if has_authentic_geo and affine and crs_str:
                geo_bbox = pixel_box_to_geographic_bbox(px_box, affine, crs_str)
                if geo_bbox:
                    w_geo, s_geo, e_geo, n_geo = geo_bbox
                    centroid_geo = [
                        round((w_geo + e_geo) / 2.0, 6),
                        round((s_geo + n_geo) / 2.0, 6)
                    ]
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [[[w_geo, s_geo], [e_geo, s_geo], [e_geo, n_geo], [w_geo, n_geo], [w_geo, s_geo]]]
                    }

                cluster_area = calculate_changed_area(
                    num_changed_pixels=cluster_pixel_count,
                    crs_str=crs_str,
                    resolution=resolution,
                    center_lat=center_lat,
                    affine_transform=affine
                )
                area_m2 = cluster_area["sq_meters"]
                area_ha = cluster_area["hectares"]

            # Average score inside cluster
            score = float(np.mean(prob_map[y_slice, x_slice][cluster_mask]))

            clusters.append(
                TemporalChangeCluster(
                    cluster_id=f"cluster_{cluster_idx}",
                    label="surface_change_cluster",
                    score=round(score, 4),
                    canvas_box=canvas_box,
                    pixel_box=px_box,
                    geographic_bbox=geo_bbox,
                    geometry=geometry,
                    area_sq_meters=area_m2,
                    area_hectares=area_ha,
                    pixel_count=cluster_pixel_count,
                    centroid_geo=centroid_geo
                )
            )
            cluster_idx += 1

        # Sort clusters by area descending (or pixel count if unreferenced)
        clusters.sort(
            key=lambda c: (c.area_sq_meters if c.area_sq_meters is not None else 0.0, (c.pixel_box[2] - c.pixel_box[0]) * (c.pixel_box[3] - c.pixel_box[1])),
            reverse=True
        )
        return clusters[:20]  # Return top 20 significant clusters

    def _save_change_mask_preview(self, mask: np.ndarray, t1_id: str, t2_id: str) -> Optional[str]:
        """Saves a binary change mask visualization PNG to local storage."""
        try:
            # 8-bit mask (255 for change, 0 for background)
            mask_u8 = (mask.astype(np.uint8)) * 255
            img = Image.fromarray(mask_u8, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            
            preview_filename = f"change_mask_{t1_id}_{t2_id}_{uuid.uuid4().hex[:6]}.png"
            storage_service.save(buf.getvalue(), preview_filename)
            return f"/api/uploads/previews/{preview_filename}"
        except Exception as e:
            logger.warning(f"Could not save change mask preview: {e}")
            return None
