from typing import Any, Dict, List, Optional, Tuple
from app.services.orchestration.plan_schema import EvidenceAssessment
from app.services.models.schemas import GroundingBox, VqaResponse, GroundingResponse
from app.schemas.evidence import EvidenceItem, BoundingBox
from app.core.logging import logger


class EvidenceAssessor:
    """
    Evidence-aware assessment service. Evaluates raw model outputs, detects geometric
    degeneracies, filters redundant overlapping detections via IoU, flags weak evidence,
    and assigns honest reliability ratings without fabricating calibrated confidence.
    """

    # Meaningful detection score threshold for open-vocabulary RS detection
    DETECTION_CONFIRMED_THRESHOLD = 0.05
    DETECTION_MINIMUM_THRESHOLD = 0.01

    def assess_grounding_evidence(
        self,
        grounding_res: GroundingResponse,
        asset_id: str,
        img_dims: Tuple[int, int]
    ) -> Tuple[List[EvidenceItem], EvidenceAssessment]:
        """
        Assesses text-guided grounding boxes, applying geometric sanity checks,
        NMS deduplication, and honest quality flagging.
        """
        img_w, img_h = img_dims
        raw_boxes = grounding_res.boxes
        raw_scores = [round(float(b.score), 4) for b in raw_boxes]
        quality_flags: List[str] = []
        limitations: List[str] = [
            "Open-vocabulary zero-shot detector without domain-specific calibration",
            "Detector scores represent raw sigmoid logits, not calibrated probabilities"
        ]
        geometric_anomalies: List[str] = []

        if not raw_boxes:
            quality_flags.append("NO_REGIONS_FOUND")
            assessment = EvidenceAssessment(
                evidence_status="INSUFFICIENT",
                raw_scores=[],
                quality_flags=quality_flags,
                limitations=limitations,
                verification_method="heuristic_spatial_audit",
                confidence=None,
                bounding_box_count=0,
                geometric_anomalies=[]
            )
            return [], assessment

        # 1. Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        filtered_boxes = self._apply_nms(raw_boxes, iou_threshold=0.60)
        if len(filtered_boxes) < len(raw_boxes):
            quality_flags.append(f"DEDUPLICATED_{len(raw_boxes) - len(filtered_boxes)}_OVERLAPPING_REGIONS")

        # 2. Geometric Sanity Checks
        valid_boxes: List[GroundingBox] = []
        scene_area = float(img_w * img_h)

        for b in filtered_boxes:
            px = b.pixel_box
            w = max(0, px[2] - px[0])
            h = max(0, px[3] - px[1])
            box_area = float(w * h)
            area_ratio = box_area / scene_area if scene_area > 0 else 0.0

            if area_ratio >= 0.85:
                msg = f"Degenerate box covering {area_ratio*100:.1f}% of entire scene ignored: {px}"
                geometric_anomalies.append(msg)
                quality_flags.append("DEGENERATE_GLOBAL_BOX_EXCLUDED")
                continue

            if area_ratio < 0.0003:
                msg = f"Sub-pixel / micro-region noise covering {area_ratio*100:.3f}% of scene ignored: {px}"
                geometric_anomalies.append(msg)
                quality_flags.append("MICRO_NOISE_EXCLUDED")
                continue

            valid_boxes.append(b)

        # 3. Evidence Status Determination
        if not valid_boxes:
            quality_flags.append("ALL_CANDIDATE_BOXES_REJECTED_BY_SANITY_CHECKS")
            assessment = EvidenceAssessment(
                evidence_status="INSUFFICIENT",
                raw_scores=raw_scores,
                quality_flags=quality_flags,
                limitations=limitations,
                verification_method="heuristic_spatial_audit",
                confidence=None,
                bounding_box_count=0,
                geometric_anomalies=geometric_anomalies
            )
            return [], assessment

        max_score = max(b.score for b in valid_boxes)
        if max_score < self.DETECTION_CONFIRMED_THRESHOLD:
            evidence_status = "WEAK"
            quality_flags.append(f"WEAK_DETECTION_SCORE_MAX_{max_score:.4f}")
            limitations.append("All region scores are below nominal detection threshold (<0.05). Findings are indicative only.")
        else:
            evidence_status = "SUPPORTED"

        # 4. Construct EvidenceItems
        evidence_items: List[EvidenceItem] = []
        for i, b in enumerate(valid_boxes):
            cb = b.canvas_box  # [x_pct, y_pct, w_pct, h_pct]
            reliability = "low" if b.score < self.DETECTION_CONFIRMED_THRESHOLD else "medium"

            box_limitations = list(limitations)
            if b.score < self.DETECTION_CONFIRMED_THRESHOLD:
                box_limitations.append("Low detector logit: unverified candidate")

            item = EvidenceItem(
                id=f"ev_ground_{i+1}",
                image_id=asset_id,
                finding_id=f"grounding_{i+1}",
                evidence_type="visual_region",
                description=f"Located '{b.label}' (raw detector score: {b.score:.4f}, uncalibrated).",
                region=BoundingBox(
                    x=round(cb[0], 2),
                    y=round(cb[1], 2),
                    width=round(cb[2], 2),
                    height=round(cb[3], 2)
                ),
                reliability=reliability,
                limitations=box_limitations,
                source_step="SINGLE_GROUNDING"
            )
            evidence_items.append(item)

        assessment = EvidenceAssessment(
            evidence_status=evidence_status,
            raw_scores=raw_scores,
            quality_flags=quality_flags,
            limitations=limitations,
            verification_method="geometric_sanity_and_nms_audit",
            confidence=None,
            bounding_box_count=len(evidence_items),
            geometric_anomalies=geometric_anomalies
        )

        return evidence_items, assessment

    def assess_vqa_evidence(
        self,
        vqa_res: VqaResponse,
        asset_id: str
    ) -> EvidenceAssessment:
        """Assesses VQA answer authenticity, confidence nullability, and quality limitations."""
        raw_ans = vqa_res.answer.strip()
        quality_flags: List[str] = []
        limitations: List[str] = [
            "Generic foundation VLM checkpoint (Salesforce/blip-vqa-base) without BigEarthNet RS adaptation",
            "Autoregressive generation logits are uncalibrated per zero-fabrication calibration rules",
            "Visual question answer generated without independent spatial verification"
        ]

        if not raw_ans or raw_ans.lower() in ["unknown", "none", ""]:
            quality_flags.append("EMPTY_OR_INDETERMINATE_ANSWER")
            evidence_status = "INSUFFICIENT"
        else:
            evidence_status = "SUPPORTED"

        return EvidenceAssessment(
            evidence_status=evidence_status,
            raw_scores=[],
            quality_flags=quality_flags,
            limitations=limitations,
            verification_method="vlm_generation_audit",
            confidence=None,
            bounding_box_count=0,
            geometric_anomalies=[]
        )

    def assess_temporal_evidence(
        self,
        temporal_res: Any,
        t1_id: str,
        t2_id: str
    ) -> Tuple[List[EvidenceItem], EvidenceAssessment]:
        """
        Assesses bi-temporal change detection evidence, converting clusters into EvidenceItems
        and assigning honest evidence status based on registration quality and change ratio.
        """
        clusters = getattr(temporal_res, "change_clusters", [])
        verdict = getattr(temporal_res, "change_verdict", "NO_CHANGE_DETECTED")
        reg = getattr(temporal_res, "registration_assessment", {}) or {}
        ratio_pct = getattr(temporal_res, "change_ratio_pct", 0.0)
        changed_ha = getattr(temporal_res, "changed_area_ha", 0.0)

        quality_flags: List[str] = [f"VERDICT_{verdict}"]
        limitations: List[str] = [
            "Bi-temporal difference model identifies binary surface changes, not multi-class semantic categories",
            "Change probability logit outputs are uncalibrated per zero-fabrication calibration rules"
        ]

        if reg.get("quality_status") == "POOR":
            quality_flags.append("POOR_COREGISTRATION")
            limitations.append("High co-registration residual detected; localized edges may exhibit false change artifacts")

        if verdict == "OBSERVED_CHANGE":
            evidence_status = "SUPPORTED"
        elif verdict == "POSSIBLE_CHANGE":
            evidence_status = "WEAK"
            limitations.append("Change magnitude is marginal (<0.50%) or affected by illumination/seasonality")
        elif verdict == "INSUFFICIENT_EVIDENCE":
            evidence_status = "INSUFFICIENT"
        else:  # NO_CHANGE_DETECTED
            evidence_status = "SUPPORTED"

        evidence_items: List[EvidenceItem] = []
        raw_scores: List[float] = []
        for i, c in enumerate(clusters):
            cb = c.canvas_box
            score = float(c.score)
            raw_scores.append(round(score, 4))
            reliability = "medium" if score >= 0.50 and reg.get("quality_status") != "POOR" else "low"

            if c.area_hectares is not None and c.area_sq_meters is not None:
                area_str = f"{c.area_hectares:.2f} ha / {c.area_sq_meters:.0f} m²"
            else:
                area_str = "unreferenced pixel cluster"

            item = EvidenceItem(
                id=f"ev_temporal_{i+1}",
                image_id=t2_id,
                finding_id=c.cluster_id,
                evidence_type="visual_region",
                description=f"Surface alteration cluster ({area_str}, raw score: {score:.3f}).",
                region=BoundingBox(
                    x=round(cb[0], 2),
                    y=round(cb[1], 2),
                    width=round(cb[2], 2),
                    height=round(cb[3], 2)
                ),
                reliability=reliability,
                limitations=list(limitations),
                source_step="TEMPORAL_CHANGE"
            )
            evidence_items.append(item)

        assessment = EvidenceAssessment(
            evidence_status=evidence_status,
            raw_scores=raw_scores,
            quality_flags=quality_flags,
            limitations=limitations,
            verification_method="bitemporal_siamese_and_phase_correlation_audit",
            confidence=None,
            bounding_box_count=len(evidence_items),
            geometric_anomalies=[]
        )
        return evidence_items, assessment

    def assess_optical_sar_evidence(
        self,
        fusion_res: Any,
        opt_id: str,
        sar_id: str
    ) -> Tuple[List[EvidenceItem], EvidenceAssessment]:
        """
        Assesses cross-modal Optical + SAR evidence clusters, evaluating agreement,
        disagreement, registration status, and physical corroboration.
        """
        evidence_items: List[EvidenceItem] = []
        quality_flags: List[str] = []
        limitations: List[str] = [
            "Cross-modal fusion combines optical spectral reflection with radar microwave backscatter",
            "Radar specular reflection (dark SAR) may indicate calm water, asphalt, or smooth dry soil",
            "Optical water identification may be obscured by clouds or confused by cloud shadows"
        ]

        reg = getattr(fusion_res, "registration_assessment", None)
        reg_status = getattr(reg, "quality_status", "UNVERIFIED") if reg else "UNVERIFIED"
        quality_flags.append(f"CROSS_MODAL_REGISTRATION_{reg_status}")

        uncertainty_state = getattr(fusion_res, "uncertainty_state", "SUPPORTED_WITH_WARNINGS")
        quality_flags.append(f"UNCERTAINTY_{uncertainty_state}")

        if reg and getattr(reg, "suppress_pixel_fusion", False):
            quality_flags.append("PIXEL_FUSION_SUPPRESSED")
            limitations.append("Pixel-level mask fusion suppressed due to unverified registration")

        # 1. Add Agreement Regions
        for idx, cluster in enumerate(getattr(fusion_res, "agreement_regions", [])):
            cb = cluster.canvas_box
            evidence_items.append(
                EvidenceItem(
                    id=f"ev_agree_{idx+1}",
                    image_id=opt_id,
                    finding_id=f"f_agree_{idx+1}",
                    evidence_type="visual_region",
                    description=f"{cluster.label}: Optical spectral and SAR specular backscatter agree.",
                    region=BoundingBox(
                        x=round(cb[0], 2),
                        y=round(cb[1], 2),
                        width=round(cb[2], 2),
                        height=round(cb[3], 2)
                    ),
                    reliability="high" if reg_status == "HIGH" else "medium",
                    limitations=list(limitations),
                    source_step="OPTICAL_SAR_ANALYSIS"
                )
            )

        # 2. Add Disagreement Regions
        for idx, cluster in enumerate(getattr(fusion_res, "disagreement_regions", [])):
            cb = cluster.canvas_box
            target_id = sar_id if "SAR" in cluster.sensor else opt_id
            evidence_items.append(
                EvidenceItem(
                    id=f"ev_disagree_{idx+1}",
                    image_id=target_id,
                    finding_id=f"f_disagree_{idx+1}",
                    evidence_type="visual_region",
                    description=f"{cluster.label}: Sensor divergence ({cluster.notes}).",
                    region=BoundingBox(
                        x=round(cb[0], 2),
                        y=round(cb[1], 2),
                        width=round(cb[2], 2),
                        height=round(cb[3], 2)
                    ),
                    reliability="medium",
                    limitations=list(limitations),
                    source_step="OPTICAL_SAR_ANALYSIS"
                )
            )

        # Map uncertainty state to evidence status
        if uncertainty_state == "SUPPORTED":
            status_str = "SUFFICIENT"
        elif uncertainty_state in ["SUPPORTED_WITH_WARNINGS", "WEAK"]:
            status_str = "WEAK"
        elif uncertainty_state == "CONFLICTING":
            status_str = "CONFLICTING"
        else:
            status_str = "INSUFFICIENT"

        assessment = EvidenceAssessment(
            evidence_status=status_str,
            raw_scores=[],
            quality_flags=quality_flags,
            limitations=limitations,
            verification_method="crossmodal_nmi_and_backscatter_physical_audit",
            confidence=None,
            bounding_box_count=len(evidence_items),
            geometric_anomalies=[]
        )
        return evidence_items, assessment

    def _apply_nms(self, boxes: List[GroundingBox], iou_threshold: float = 0.60) -> List[GroundingBox]:
        """Performs greedy Non-Maximum Suppression to remove redundant overlapping boxes."""
        if not boxes:
            return []

        # Sort descending by score
        sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
        keep: List[GroundingBox] = []

        for candidate in sorted_boxes:
            should_keep = True
            for kept in keep:
                iou = self._calculate_iou(candidate.pixel_box, kept.pixel_box)
                if iou >= iou_threshold and candidate.label == kept.label:
                    should_keep = False
                    break
            if should_keep:
                keep.append(candidate)

        return keep

    def _calculate_iou(self, boxA: List[int], boxB: List[int]) -> float:
        """Calculates Intersection over Union of two [x1, y1, x2, y2] boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = float(interW * interH)

        areaA = float(max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1]))
        areaB = float(max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1]))

        unionArea = areaA + areaB - interArea
        if unionArea <= 0:
            return 0.0
        return interArea / unionArea
