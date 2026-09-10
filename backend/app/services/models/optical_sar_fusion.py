import os
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import rasterio

from app.domain.tasks import TaskFamily
from app.domain.modalities import Modality
from app.schemas.image_asset import ImageAsset
from app.db.asset_repository import asset_repository
from app.services.storage import storage_service
from app.services.geospatial.sar_processing import preprocess_sar_raster, SarProcessingReport
from app.services.geospatial.cross_modal_alignment import (
    verify_cross_modal_registration,
    CrossModalRegistrationAssessment
)
from app.core.logging import logger


class OpticalSarFusionRequest(BaseModel):
    optical_asset_id: str
    sar_asset_id: str
    query: str = "Analyze complementary optical and SAR features"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class OpticalSarEvidenceCluster(BaseModel):
    id: str
    label: str
    sensor: str  # OPTICAL, SAR, FUSED
    status: str  # AGREEMENT, DISAGREEMENT, SENSOR_SPECIFIC
    score: float
    canvas_box: List[float]  # [x, y, width, height] in 0-100 percentage
    pixel_box: List[int]     # [ymin, xmin, ymax, xmax]
    geometry: Optional[Dict[str, Any]] = None
    area_sq_meters: Optional[float] = None
    notes: str = ""


class OpticalSarFusionResponse(BaseModel):
    model_name: str = "SatQuery-OpticalSAR-FusionAdapter-v1"
    task_family: TaskFamily = TaskFamily.OPTICAL_SAR_ANALYSIS
    direct_answer: str
    uncertainty_state: str  # SUPPORTED, SUPPORTED_WITH_WARNINGS, WEAK, CONFLICTING, INSUFFICIENT_EVIDENCE, OUT_OF_DOMAIN, UNAVAILABLE
    confidence: Optional[float] = None  # None per SIH PS 26167
    optical_evidence: List[OpticalSarEvidenceCluster] = Field(default_factory=list)
    sar_evidence: List[OpticalSarEvidenceCluster] = Field(default_factory=list)
    agreement_regions: List[OpticalSarEvidenceCluster] = Field(default_factory=list)
    disagreement_regions: List[OpticalSarEvidenceCluster] = Field(default_factory=list)
    missing_evidence_regions: List[OpticalSarEvidenceCluster] = Field(default_factory=list)
    fused_findings: List[Dict[str, Any]] = Field(default_factory=list)
    registration_assessment: CrossModalRegistrationAssessment
    sar_processing_report: SarProcessingReport
    optical_water_ratio_pct: float = 0.0
    sar_low_backscatter_ratio_pct: float = 0.0
    agreement_ratio_pct: float = 0.0
    disagreement_ratio_pct: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class OpticalSarFusionAdapter:
    """
    Scientifically valid, evidence-first Optical + SAR Cross-Modal Fusion Adapter.
    Never feeds SAR imagery into optical RGB models.
    Operates on physical sensor evidence: optical spectral indices + calibrated SAR backscatter.
    """
    def __init__(self):
        self.model_name = "SatQuery-OpticalSAR-FusionAdapter-v1"
        self.task_family = TaskFamily.OPTICAL_SAR_ANALYSIS
        self.supported_modalities = [Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR]

    def predict(self, request: OpticalSarFusionRequest) -> OpticalSarFusionResponse:
        t0 = time.time()
        warnings: List[str] = []
        limitations: List[str] = []

        # 1. Asset Retrieval & Modality Validation
        opt_asset = asset_repository.get_asset(request.optical_asset_id)
        sar_asset = asset_repository.get_asset(request.sar_asset_id)

        if not opt_asset or not sar_asset:
            duration = (time.time() - t0) * 1000
            return OpticalSarFusionResponse(
                direct_answer="Failed to load input assets from repository.",
                uncertainty_state="UNAVAILABLE",
                registration_assessment=CrossModalRegistrationAssessment(),
                sar_processing_report=SarProcessingReport(asset_id="unknown"),
                warnings=["One or both asset IDs could not be resolved."],
                limitations=["Input data missing"],
                latency_ms=duration
            )

        # Ensure correct assignment if user provided them in reverse order
        if opt_asset.modality == Modality.SAR and sar_asset.modality in [Modality.OPTICAL, Modality.MULTISPECTRAL]:
            opt_asset, sar_asset = sar_asset, opt_asset

        if sar_asset.modality != Modality.SAR:
            warnings.append(f"Asset '{sar_asset.id}' is not verified SAR modality.")
        if opt_asset.modality not in [Modality.OPTICAL, Modality.MULTISPECTRAL]:
            warnings.append(f"Asset '{opt_asset.id}' is not optical/multispectral modality.")

        # 2. Read rasters into memory
        opt_path = opt_asset.storage_path if (os.path.isabs(opt_asset.storage_path) and os.path.exists(opt_asset.storage_path)) else storage_service.get_full_path(os.path.basename(opt_asset.storage_path))
        sar_path = sar_asset.storage_path if (os.path.isabs(sar_asset.storage_path) and os.path.exists(sar_asset.storage_path)) else storage_service.get_full_path(os.path.basename(sar_asset.storage_path))

        with rasterio.open(opt_path) as src_opt:
            opt_data = src_opt.read()
        with rasterio.open(sar_path) as src_sar:
            sar_data = src_sar.read()

        # 3. Independent SAR Preprocessing (Enhanced Lee Filter + sigma0 in dB)
        sar_db, sar_report = preprocess_sar_raster(sar_data, sar_asset, apply_speckle_filter=True)
        warnings.extend(sar_report.warnings)

        # 4. Cross-Modal Registration Gate
        reg_assessment = verify_cross_modal_registration(opt_data, sar_db, opt_asset, sar_asset)
        warnings.extend(reg_assessment.warnings)
        limitations.extend(reg_assessment.limitations)

        # 5. Independent Sensor Specialists
        # 5a. Optical Specialist: Extract optical water / low-reflectance candidates
        opt_h, opt_w = opt_data.shape[1], opt_data.shape[2]
        if opt_data.shape[0] >= 3:
            # Green (band 2) vs NIR or Blue
            green = opt_data[1].astype(np.float32)
            red = opt_data[0].astype(np.float32)
            blue = opt_data[2].astype(np.float32)
            luma = 0.299 * red + 0.587 * green + 0.114 * blue
            # Water in optical is typically low luma and darker than surrounding land
            opt_water_mask = (luma <= np.percentile(luma, 35)) & (luma < np.mean(luma))
        else:
            opt_2d = opt_data[0].astype(np.float32)
            opt_water_mask = (opt_2d <= np.percentile(opt_2d, 35)) & (opt_2d < np.mean(opt_2d))

        # 5b. SAR Specialist: Calibrated backscatter thresholding
        # Water exhibits specular reflection -> very low backscatter (sigma0 < -15 dB or lowest 20%)
        sar_water_threshold_db = float(request.parameters.get("sar_db_threshold", -15.0))
        sar_water_mask = (sar_db <= sar_water_threshold_db) & (sar_db > -45.0)

        # Fallback if uncalibrated amplitude or uniform backscatter: use lowest quartile
        if np.sum(sar_water_mask) < 20:
            sar_water_mask = (sar_db <= np.percentile(sar_db, 25)) & (sar_db < np.mean(sar_db))

        # Urban/built structures exhibit double-bounce -> high backscatter (sigma0 > -6 dB)
        sar_urban_mask = sar_db > -6.0

        # Resample/Crop to mutual minimum grid for spatial comparison
        min_h = min(opt_water_mask.shape[0], sar_water_mask.shape[0])
        min_w = min(opt_water_mask.shape[1], sar_water_mask.shape[1])

        opt_w_sub = opt_water_mask[:min_h, :min_w]
        sar_w_sub = sar_water_mask[:min_h, :min_w]
        sar_u_sub = sar_urban_mask[:min_h, :min_w]

        total_mutual_px = min_h * min_w
        opt_water_ratio = round(float(np.sum(opt_w_sub) / total_mutual_px * 100.0), 2)
        sar_water_ratio = round(float(np.sum(sar_w_sub) / total_mutual_px * 100.0), 2)

        # 6. Complementary Spatial Evidence Comparison
        agreement_clusters: List[OpticalSarEvidenceCluster] = []
        disagreement_clusters: List[OpticalSarEvidenceCluster] = []
        optical_clusters: List[OpticalSarEvidenceCluster] = []
        sar_clusters: List[OpticalSarEvidenceCluster] = []
        fused_findings: List[Dict[str, Any]] = []

        agreement_ratio = 0.0
        disagreement_ratio = 0.0

        if reg_assessment.suppress_pixel_fusion:
            # Registration is POOR or UNVERIFIED: Suppress pixel-level mask fusion
            limitations.append(
                "Pixel-level spatial overlay was suppressed due to unverified or poor cross-modal registration. "
                "Observations are reported at scene level only."
            )
            uncertainty_state = "INSUFFICIENT_EVIDENCE" if reg_assessment.quality_status == "POOR" else "SUPPORTED_WITH_WARNINGS"

            # Coarse sensor findings
            fused_findings.append({
                "type": "coarse_complementary_observation",
                "optical_summary": f"Optical sensor indicates ~{opt_water_ratio}% low-reflectance / water candidates.",
                "sar_summary": f"SAR backscatter indicates ~{sar_water_ratio}% specular reflection (< -15 dB).",
                "verdict": "UNVERIFIED_SPATIAL_ALIGNMENT",
                "notes": "Direct pixel-by-pixel intersection omitted."
            })

            answer = (
                f"Cross-modal analysis completed with unverified registration ({reg_assessment.quality_status}). "
                f"Optical scene exhibits {opt_water_ratio}% candidate water features; SAR indicates {sar_water_ratio}% "
                f"specular low-backscatter surfaces. Direct spatial overlay was suppressed per SIH PS 26167 safeguards."
            )

        else:
            # Verified registration: Compute spatial agreement and disagreement
            agreement_mask = opt_w_sub & sar_w_sub
            # Disagreement 1: Optical indicates water, but SAR exhibits high/rough backscatter
            disagree_opt_only = opt_w_sub & (~sar_w_sub)
            # Disagreement 2: SAR indicates specular water, but optical is not identified as water (e.g. clouds or sediment)
            disagree_sar_only = sar_w_sub & (~opt_w_sub)

            agree_px = int(np.sum(agreement_mask))
            disagree_px = int(np.sum(disagree_opt_only) + np.sum(disagree_sar_only))

            agreement_ratio = round(float(agree_px / total_mutual_px * 100.0), 2)
            disagreement_ratio = round(float(disagree_px / total_mutual_px * 100.0), 2)

            # Generate vector clusters
            agreement_clusters = self._extract_clusters(
                agreement_mask, "AGREEMENT", "Fused Water Body (Dual-Sensor Confirmed)",
                min_h, min_w, opt_asset
            )
            disagree_opt_clusters = self._extract_clusters(
                disagree_opt_only, "DISAGREEMENT", "Optical-Only Water (Unconfirmed by SAR Backscatter)",
                min_h, min_w, opt_asset
            )
            disagree_sar_clusters = self._extract_clusters(
                disagree_sar_only, "DISAGREEMENT", "SAR Specular Surface (Potential Cloud Penetration / Smooth Terrain)",
                min_h, min_w, opt_asset
            )
            disagreement_clusters = disagree_opt_clusters + disagree_sar_clusters

            # Sensor-specific clusters
            optical_clusters = self._extract_clusters(
                opt_w_sub, "SENSOR_SPECIFIC", "Optical Spectral Water Candidate",
                min_h, min_w, opt_asset, max_clusters=3
            )
            sar_clusters = self._extract_clusters(
                sar_w_sub, "SENSOR_SPECIFIC", "SAR Specular Reflection Candidate",
                min_h, min_w, sar_asset, max_clusters=3
            )

            # 7. Uncertainty State Assignment
            if agreement_ratio > 1.0 and disagreement_ratio < (agreement_ratio * 1.5):
                uncertainty_state = "SUPPORTED" if reg_assessment.quality_status == "HIGH" else "SUPPORTED_WITH_WARNINGS"
            elif disagreement_ratio > 3.0 and agreement_ratio < 0.5:
                uncertainty_state = "CONFLICTING"
                warnings.append("High cross-sensor disagreement: Optical and SAR signatures diverge substantially across footprint.")
            elif agreement_ratio < 0.2 and opt_water_ratio < 0.5 and sar_water_ratio < 0.5:
                uncertainty_state = "WEAK"
                warnings.append("Minimal water or distinct backscatter targets identified across mutual footprint.")
            else:
                uncertainty_state = "SUPPORTED_WITH_WARNINGS"

            # Construct findings list
            fused_findings.append({
                "type": "cross_modal_agreement",
                "summary": f"Dual-sensor confirmed water: {agreement_ratio}% of mutual footprint ({len(agreement_clusters)} cluster(s)).",
                "optical_support": f"{opt_water_ratio}% candidate water pixels",
                "sar_support": f"{sar_water_ratio}% specular backscatter pixels",
                "quality": uncertainty_state
            })

            if disagreement_clusters:
                fused_findings.append({
                    "type": "cross_modal_disagreement",
                    "summary": f"Sensor divergence detected across {disagreement_ratio}% of footprint.",
                    "notes": "May indicate cloud penetration by SAR, atmospheric haze, surface roughness, or specular non-water surfaces.",
                    "quality": "CONFLICTING" if uncertainty_state == "CONFLICTING" else "SUPPORTED_WITH_WARNINGS"
                })

            answer = (
                f"Cross-modal analysis confirmed {agreement_ratio}% mutual water coverage where optical spectral absorption "
                f"and SAR specular low backscatter (< {sar_water_threshold_db} dB) physically corroborate. "
                f"Sensor divergence noted across {disagreement_ratio}% of the scene (Uncertainty: {uncertainty_state})."
            )

        # Strict SIH rule: No synthetic confidence
        confidence = None

        duration_ms = round((time.time() - t0) * 1000, 2)
        return OpticalSarFusionResponse(
            direct_answer=answer,
            uncertainty_state=uncertainty_state,
            confidence=confidence,
            optical_evidence=optical_clusters,
            sar_evidence=sar_clusters,
            agreement_regions=agreement_clusters,
            disagreement_regions=disagreement_clusters,
            missing_evidence_regions=[],
            fused_findings=fused_findings,
            registration_assessment=reg_assessment,
            sar_processing_report=sar_report,
            optical_water_ratio_pct=opt_water_ratio,
            sar_low_backscatter_ratio_pct=sar_water_ratio,
            agreement_ratio_pct=agreement_ratio,
            disagreement_ratio_pct=disagreement_ratio,
            warnings=warnings,
            limitations=limitations,
            latency_ms=duration_ms
        )

    def _extract_clusters(
        self,
        binary_mask: np.ndarray,
        status: str,
        label: str,
        h: int,
        w: int,
        asset: ImageAsset,
        max_clusters: int = 5
    ) -> List[OpticalSarEvidenceCluster]:
        """Extract bounding boxes and polygon geometry from connected components."""
        from scipy.ndimage import label as nd_label
        clusters: List[OpticalSarEvidenceCluster] = []

        labeled, num_features = nd_label(binary_mask)
        if num_features == 0:
            return clusters

        # Find largest components
        sizes = [int(np.sum(labeled == i)) for i in range(1, num_features + 1)]
        ranked_indices = np.argsort(sizes)[::-1][:max_clusters]

        for rank, idx in enumerate(ranked_indices):
            comp_id = idx + 1
            px_count = sizes[idx]
            if px_count < 25:  # Minimum 25 px cluster
                continue

            ys, xs = np.where(labeled == comp_id)
            ymin, ymax = int(np.min(ys)), int(np.max(ys))
            xmin, xmax = int(np.min(xs)), int(np.max(xs))

            # Canvas Box (0-100%)
            c_x = round(float(xmin / w * 100.0), 2)
            c_y = round(float(ymin / h * 100.0), 2)
            c_w = round(float(max(1, xmax - xmin) / w * 100.0), 2)
            c_h = round(float(max(1, ymax - ymin) / h * 100.0), 2)

            # Geographic Area if authentic CRS exists
            area_m2 = None
            if asset.resolution and asset.crs:
                area_m2 = round(float(px_count * (asset.resolution ** 2)), 2)

            # Simple bounding box GeoJSON geometry
            geometry = None
            if asset.affine_transform and asset.crs:
                try:
                    from rasterio.transform import xy
                    t = asset.affine_transform
                    aff = rasterio.Affine(t[0], t[1], t[2], t[3], t[4], t[5])
                    gx_min, gy_min = xy(aff, ymax, xmin)
                    gx_max, gy_max = xy(aff, ymin, xmax)
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [[[gx_min, gy_min], [gx_max, gy_min], [gx_max, gy_max], [gx_min, gy_max], [gx_min, gy_min]]]
                    }
                except Exception:
                    pass

            clusters.append(
                OpticalSarEvidenceCluster(
                    id=f"{status.lower()}_{rank+1}",
                    label=f"{label} #{rank+1}",
                    sensor="FUSED" if status == "AGREEMENT" else ("OPTICAL" if "Optical" in label else "SAR"),
                    status=status,
                    score=0.85 if status == "AGREEMENT" else 0.60,
                    canvas_box=[c_x, c_y, c_w, c_h],
                    pixel_box=[ymin, xmin, ymax, xmax],
                    geometry=geometry,
                    area_sq_meters=area_m2,
                    notes=f"{px_count} pixels"
                )
            )

        return clusters


optical_sar_fusion_adapter = OpticalSarFusionAdapter()
