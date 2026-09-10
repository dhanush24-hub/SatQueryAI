import re
import time
from typing import Any, Dict, List, Optional
from app.domain.modalities import Modality
from app.domain.tasks import TaskFamily
from app.services.models.base import ModelAdapter
from app.services.models.schemas import (
    TemporalVqaRequest,
    TemporalVqaResponse,
    TemporalChangeRequest,
    TemporalChangeResponse,
    TemporalChangeCluster
)
from app.services.models.temporal_change import TemporalChangeAdapter
from app.registry.model_registry import model_tool_registry
from app.core.logging import logger


class TemporalVqaAdapter(ModelAdapter):
    """
    Specialist bi-temporal visual question answering adapter.
    Answers natural language change inquiries grounded strictly in verified
    empirical change evidence, with transparent semantic boundaries.
    """

    MODEL_ID = "satquery/temporal-change-vqa-engine"

    def __init__(self, change_adapter: Optional[TemporalChangeAdapter] = None):
        self._change_adapter = change_adapter or TemporalChangeAdapter()

    @property
    def model_name(self) -> str:
        return self.MODEL_ID

    @property
    def task_family(self) -> TaskFamily:
        return TaskFamily.TEMPORAL_CHANGE_VQA

    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.OPTICAL, Modality.MULTISPECTRAL]

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def health(self) -> Dict[str, Any]:
        return {"status": "ready", "model_name": self.model_name}

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task_family.value,
            "supported_modalities": [m.value for m in self.supported_modalities],
            "description": "Grounded bi-temporal VQA deriving answers from verified change masks and spatial clusters."
        }

    def predict(self, request: TemporalVqaRequest) -> TemporalVqaResponse:
        t0 = time.time()

        # 1. Execute or obtain underlying temporal change detection
        method = request.parameters.get("method", "siamunet_diff")
        threshold = float(request.parameters.get("threshold", 0.50))
        change_req = TemporalChangeRequest(
            t1_asset_id=request.t1_asset_id,
            t2_asset_id=request.t2_asset_id,
            method=method,
            threshold=threshold,
            parameters=request.parameters
        )
        change_res = self._change_adapter.predict(change_req)

        # 2. Extract key physical evidence parameters
        verdict = change_res.change_verdict
        changed_ha = change_res.changed_area_ha
        changed_m2 = change_res.changed_area_m2
        ratio_pct = change_res.change_ratio_pct
        clusters = change_res.change_clusters
        meta = change_res.metadata.get("temporal_pair", {})
        t1_time = meta.get("t1_acquisition_time") or "T1"
        t2_time = meta.get("t2_acquisition_time") or "T2"
        reg_meta = change_res.registration_assessment or {}

        warnings = list(change_res.warnings)
        warnings.append("Confidence is reported as null: Uncalibrated change VQA reasoning.")

        # 3. Interpret user question semantics
        q_lower = request.question.lower()
        is_semantic_specific = any(w in q_lower for w in ["built-up", "building", "buildings", "water", "forest", "vegetation", "deforestation", "urban"])
        is_location_query = any(w in q_lower for w in ["where", "which region", "which area", "locate", "perimeter", "quadrant"])

        # 4. Synthesize evidence-grounded answer
        if verdict == "INSUFFICIENT_EVIDENCE":
            answer = (
                f"Temporal change inquiry cannot be reliably answered due to co-registration uncertainty "
                f"between {t1_time} and {t2_time}. Spatial shift residual ({reg_meta.get('shift_magnitude_pixels', 0.0)} px) "
                f"exceeds tolerance, preventing verified pixel-level change attribution."
            )

        elif verdict == "NO_CHANGE_DETECTED":
            answer = (
                f"No significant land-cover change detected between {t1_time} and {t2_time}. "
                f"The observed change ratio is {ratio_pct:.3f}% (less than the nominal detection threshold of 0.05%), "
                f"indicating spatial surface stability across the mutually overlapping footprint."
            )

        else:
            # OBSERVED_CHANGE or POSSIBLE_CHANGE
            top_cluster = clusters[0] if clusters else None
            cluster_summary = ""
            if top_cluster:
                loc_desc = self._describe_cluster_location(top_cluster)
                if top_cluster.area_hectares is not None:
                    cluster_summary = f" The largest contiguous change cluster spans {top_cluster.area_hectares:.2f} ha ({top_cluster.area_sq_meters:.0f} m²) {loc_desc}."
                else:
                    cluster_summary = f" The largest contiguous change cluster {loc_desc}."

            area_extent_str = f"{changed_ha:.2f} hectares" if changed_ha is not None else f"{change_res.total_changed_pixels:,} pixels"

            if is_semantic_specific:
                # Strictly enforce scientific boundaries: do NOT fabricate semantic labels without a land-cover classifier
                matched_category = "requested semantic class"
                for cat in ["built-up", "water", "forest", "vegetation", "buildings"]:
                    if cat in q_lower:
                        matched_category = cat
                        break

                answer = (
                    f"Surface alteration was detected across {area_extent_str} ({ratio_pct:.2f}% of the scene) "
                    f"between {t1_time} and {t2_time}.{cluster_summary} "
                    f"However, because the bi-temporal change detector identifies spatial difference without a multi-class "
                    f"semantic land-cover segmenter, specific categorical transitions (such as whether this alteration represents "
                    f"an increase/decrease in {matched_category}) cannot be conclusively confirmed from binary change masks alone. "
                    f"Ground alteration is verified at the indicated cluster coordinates."
                )
                warnings.append(
                    f"Semantic boundary limitation: Binary change detection cannot definitively assign '{matched_category}' "
                    "label without multi-temporal semantic land-cover classification."
                )

            elif is_location_query:
                loc_details = []
                for c in clusters[:3]:
                    loc_desc = self._describe_cluster_location(c)
                    if c.area_hectares is not None:
                        loc_details.append(f"{c.cluster_id}: {c.area_hectares:.2f} ha {loc_desc}")
                    else:
                        loc_details.append(f"{c.cluster_id}: {loc_desc}")
                details_str = "; ".join(loc_details)
                answer = (
                    f"Significant surface changes between {t1_time} and {t2_time} are localized across {len(clusters)} "
                    f"primary cluster(s) totaling {area_extent_str} ({ratio_pct:.2f}% of scene). "
                    f"Locations: {details_str}."
                )

            else:
                answer = (
                    f"Between {t1_time} and {t2_time}, surface change was detected across {area_extent_str} "
                    f"({ratio_pct:.2f}% of the scene). A total of {len(clusters)} spatial change cluster(s) were delineated.{cluster_summary}"
                )

        duration_ms = (time.time() - t0) * 1000.0

        return TemporalVqaResponse(
            task_family=TaskFamily.TEMPORAL_CHANGE_VQA,
            model_name=self.model_name,
            device=change_res.device,
            inference_latency_ms=round(duration_ms, 1),
            answer=answer,
            change_verdict=verdict,
            change_clusters=clusters,
            changed_area_ha=changed_ha,
            change_ratio_pct=ratio_pct,
            total_changed_pixels=change_res.total_changed_pixels,
            has_authentic_geo=change_res.has_authentic_geo,
            mask_preview_url=change_res.mask_preview_url,
            registration_assessment=change_res.registration_assessment,
            confidence=None,
            warnings=warnings,
            metadata={
                "change_model": change_res.model_name,
                "total_changed_pixels": change_res.total_changed_pixels,
                "registration": reg_meta
            }
        )

    def _describe_cluster_location(self, cluster: TemporalChangeCluster) -> str:
        """Derives natural geographic or quadrant description from canvas/geo bounding box."""
        cb = cluster.canvas_box  # [x_min_pct, y_min_pct, width_pct, height_pct]
        cx = cb[0] + cb[2] / 2.0
        cy = cb[1] + cb[3] / 2.0

        vert = "central"
        if cy < 35.0:
            vert = "northern"
        elif cy > 65.0:
            vert = "southern"

        horiz = "central"
        if cx < 35.0:
            horiz = "western"
        elif cx > 65.0:
            horiz = "eastern"

        if vert == "central" and horiz == "central":
            sector = "in the central sector"
        elif vert == "central":
            sector = f"in the {horiz} sector"
        elif horiz == "central":
            sector = f"in the {vert} sector"
        else:
            sector = f"in the {vert}-{horiz} sector"

        if cluster.centroid_geo:
            return f"{sector} (centroid: {cluster.centroid_geo[1]:.4f}°N, {cluster.centroid_geo[0]:.4f}°E)"
        return sector
