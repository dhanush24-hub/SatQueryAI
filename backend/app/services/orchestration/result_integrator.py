import uuid
from typing import Any, Dict, List, Optional
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.schemas.analysis import (
    AnalysisResult,
    FindingItem,
    OverlayData,
    RegistrationInfo,
    DomainSuitabilityInfo,
    ModelProvenanceInfo,
    DownloadableArtifacts
)
from app.schemas.evidence import EvidenceItem, BoundingBox
from app.schemas.execution import ExecutionStep
from app.schemas.image_asset import ImageAsset
from app.services.orchestration.plan_schema import WorkflowPlan, EvidenceAssessment
from app.services.orchestration.evidence_assessor import EvidenceAssessor
from app.services.models.schemas import VqaResponse, GroundingResponse
from app.core.logging import logger


class ResultIntegrator:
    """
    Synthesizes workflow outputs, evidence assessments, and execution telemetry
    into a structured, observable AnalysisResult with strict evidence-first guarantees.
    """

    def __init__(self):
        self.evidence_assessor = EvidenceAssessor()

    def integrate(
        self,
        plan: WorkflowPlan,
        step_outputs: Dict[str, Any],
        execution_trace: List[ExecutionStep],
        assets: List[ImageAsset],
        error_msg: Optional[str] = None,
        analysis_id: Optional[str] = None
    ) -> AnalysisResult:
        analysis_id = analysis_id or f"anl_{uuid.uuid4().hex[:12]}"
        primary_asset = assets[0] if assets else None
        asset_id = primary_asset.id if primary_asset else ""
        img_dims = (primary_asset.width, primary_asset.height) if primary_asset else (256, 256)

        warnings: List[str] = []
        evidence_items: List[EvidenceItem] = []
        findings_items: List[FindingItem] = []
        answer_parts: List[str] = []
        assessment: Optional[EvidenceAssessment] = None

        # Handle execution error
        if error_msg:
            return AnalysisResult(
                id=analysis_id,
                answer=f"Analysis failed: {error_msg}",
                task=plan.task_family,
                confidence=None,
                evidence=[],
                findings=[],
                warnings=[error_msg],
                execution_summary=execution_trace,
                status=AnalysisStatus.FAILED,
                evidence_state="UNAVAILABLE",
                evidence_assessment=EvidenceAssessment(
                    evidence_status="UNAVAILABLE",
                    raw_scores=[],
                    quality_flags=["STEP_EXECUTION_FAILURE"],
                    limitations=["Workflow execution halted prematurely"],
                    verification_method="none",
                    confidence=None
                ),
                downloadable_artifacts=DownloadableArtifacts(
                    report_pdf_url=f"/api/analysis/{analysis_id}/report.pdf",
                    result_json_url=f"/api/analysis/{analysis_id}/export.json"
                )
            )

        # 1. Process VQA outputs
        vqa_out: Optional[VqaResponse] = None
        for step in plan.steps:
            if step.task_family == TaskFamily.SINGLE_VQA and step.step_id in step_outputs:
                vqa_out = step_outputs[step.step_id]
                break

        if vqa_out:
            vqa_ans = vqa_out.answer.strip()
            answer_parts.append(f"Answer: {vqa_ans}")
            vqa_assessment = self.evidence_assessor.assess_vqa_evidence(vqa_out, asset_id)
            warnings.extend(vqa_out.warnings)
            warnings.append(
                "Confidence is reported as null: Autoregressive VLM generation logits are uncalibrated per SIH PS 26167 rules."
            )
            assessment = vqa_assessment
            findings_items.append(
                FindingItem(
                    finding_id="f_vqa_1",
                    type="vqa_observation",
                    summary=vqa_ans,
                    evidence_quality="SUPPORTED" if vqa_assessment.evidence_status == "SUFFICIENT" else "SUPPORTED_WITH_WARNINGS",
                    source_model=vqa_out.model_name,
                    limitations=["Single-image visual interpretation without multi-spectral confirmation"]
                )
            )

        # 2. Process Grounding outputs
        grounding_out: Optional[GroundingResponse] = None
        for step in plan.steps:
            if step.task_family == TaskFamily.SINGLE_GROUNDING and step.step_id in step_outputs:
                grounding_out = step_outputs[step.step_id]
                break

        if grounding_out:
            items, ground_assessment = self.evidence_assessor.assess_grounding_evidence(
                grounding_out, asset_id, img_dims
            )
            evidence_items.extend(items)
            assessment = ground_assessment  # Grounding assessment takes precedence for spatial tasks
            warnings.extend(grounding_out.warnings)

            if ground_assessment.evidence_status == "INSUFFICIENT":
                answer_parts.append("No spatial regions could be reliably grounded matching the instruction.")
                warnings.append("Localization abstained: No candidate regions met geometric sanity and score thresholds.")
            elif ground_assessment.evidence_status == "WEAK":
                answer_parts.append(
                    f"Located {len(items)} tentative candidate region(s). Detector scores are weak (<0.05)."
                )
                warnings.append("Caution: Grounding scores are low. Highlighted regions should be treated as indicative candidates.")
            else:
                answer_parts.append(f"Grounded {len(items)} spatial region(s) matching the target query.")

            for g_idx, box in enumerate(grounding_out.boxes):
                findings_items.append(
                    FindingItem(
                        finding_id=f"f_ground_{g_idx+1}",
                        type="single_grounding",
                        summary=f"Spatial detection: {box.label} (score: {box.score:.2f})",
                        bbox_px=box.pixel_box,
                        canvas_box=BoundingBox(
                            x=box.canvas_box[0],
                            y=box.canvas_box[1],
                            width=box.canvas_box[2],
                            height=box.canvas_box[3]
                        ),
                        evidence_quality="SUPPORTED" if box.score >= 0.10 else "WEAK",
                        source_model=grounding_out.model_name,
                        limitations=["Detection boundary subject to optical patch receptive field limitations"]
                    )
                )

        # 3. Process Temporal Change outputs
        temp_out: Optional[Any] = None
        for step in plan.steps:
            if step.task_family == TaskFamily.TEMPORAL_CHANGE and step.step_id in step_outputs:
                temp_out = step_outputs[step.step_id]
                break

        if temp_out:
            t1_id = assets[0].id if len(assets) > 0 else ""
            t2_id = assets[1].id if len(assets) > 1 else ""
            items, temp_assessment = self.evidence_assessor.assess_temporal_evidence(temp_out, t1_id, t2_id)
            evidence_items.extend(items)
            assessment = temp_assessment
            warnings.extend(getattr(temp_out, "warnings", []))

            verdict = getattr(temp_out, "change_verdict", "NO_CHANGE_DETECTED")
            area_ha = getattr(temp_out, "changed_area_ha", None)
            ratio_pct = getattr(temp_out, "change_ratio_pct", 0.0)
            has_geo = getattr(temp_out, "has_authentic_geo", False)
            num_clusters = len(items)

            if verdict == "NO_CHANGE_DETECTED":
                answer_parts.append("No significant land-cover change detected across the mutual footprint (change ratio < 0.05%).")
            elif verdict == "INSUFFICIENT_EVIDENCE":
                answer_parts.append("Temporal change detection was inconclusive due to high co-registration uncertainty.")
            else:
                if has_geo and area_ha is not None:
                    answer_parts.append(
                        f"Detected {verdict}: {area_ha:.2f} ha ({ratio_pct:.2f}% of scene) across {num_clusters} spatial cluster(s)."
                    )
                else:
                    answer_parts.append(
                        f"Detected {verdict}: {ratio_pct:.2f}% of pixel area across {num_clusters} spatial cluster(s). Physical area omitted due to unreferenced coordinate grid."
                    )



        # 4. Process Temporal Change VQA outputs
        temp_vqa_out: Optional[Any] = None
        for step in plan.steps:
            if step.task_family == TaskFamily.TEMPORAL_CHANGE_VQA and step.step_id in step_outputs:
                temp_vqa_out = step_outputs[step.step_id]
                break

        if temp_vqa_out:
            t1_id = assets[0].id if len(assets) > 0 else ""
            t2_id = assets[1].id if len(assets) > 1 else ""
            items, temp_vqa_assessment = self.evidence_assessor.assess_temporal_evidence(temp_vqa_out, t1_id, t2_id)
            evidence_items.extend(items)
            assessment = temp_vqa_assessment
            warnings.extend(getattr(temp_vqa_out, "warnings", []))
            answer_parts.append(getattr(temp_vqa_out, "answer", ""))

        # Build FindingItem objects from change clusters (from either TEMPORAL_CHANGE or TEMPORAL_CHANGE_VQA)
        temporal_obj = temp_out or temp_vqa_out
        if temporal_obj:
            clusters = getattr(temporal_obj, "change_clusters", [])
            for c_idx, cluster in enumerate(clusters):
                c_box = cluster.canvas_box
                canvas_b = BoundingBox(x=c_box[0], y=c_box[1], width=c_box[2], height=c_box[3])
                findings_items.append(
                    FindingItem(
                        finding_id=f"f_temp_{c_idx+1}",
                        type="observed_change",
                        summary=f"Surface change cluster {c_idx+1} (mean prob: {cluster.score:.2f})",
                        bbox_px=cluster.pixel_box,
                        canvas_box=canvas_b,
                        geometry=cluster.geometry,
                        changed_pixels=getattr(cluster, "pixel_count", None),
                        area_m2=cluster.area_sq_meters,
                        evidence_quality="SUPPORTED",
                        source_model=getattr(temporal_obj, "model_name", "AttentionChangeNet"),
                        limitations=["Binary structural change detection; land-use semantics not differentiated"]
                    )
                )

        # 5. Process Optical + SAR Fusion outputs
        fusion_out: Optional[Any] = None
        for step in plan.steps:
            if step.task_family == TaskFamily.OPTICAL_SAR_ANALYSIS and step.step_id in step_outputs:
                fusion_out = step_outputs[step.step_id]
                break

        if fusion_out:
            opt_id = assets[0].id if len(assets) > 0 else ""
            sar_id = assets[1].id if len(assets) > 1 else ""
            items, fusion_assessment = self.evidence_assessor.assess_optical_sar_evidence(
                fusion_out, opt_id, sar_id
            )
            evidence_items.extend(items)
            assessment = fusion_assessment
            warnings.extend(getattr(fusion_out, "warnings", []))
            answer_parts.append(getattr(fusion_out, "direct_answer", ""))

            # Build FindingItem objects for fused findings
            for idx, cluster in enumerate(getattr(fusion_out, "agreement_regions", [])):
                cb = cluster.canvas_box
                canvas_b = BoundingBox(x=cb[0], y=cb[1], width=cb[2], height=cb[3])
                findings_items.append(
                    FindingItem(
                        finding_id=f"f_agree_{idx+1}",
                        type="cross_modal_agreement",
                        summary=f"{cluster.label}: Dual-sensor confirmed water (spectral absorption & specular backscatter).",
                        bbox_px=cluster.pixel_box,
                        canvas_box=canvas_b,
                        geometry=cluster.geometry,
                        area_m2=cluster.area_sq_meters,
                        evidence_quality="SUPPORTED",
                        source_model=getattr(fusion_out, "model_name", "SatQuery-OpticalSAR-FusionAdapter-v1"),
                        limitations=["Corroborated surface smoothness and absorption; land cover semantics remain coarse"]
                    )
                )

            for idx, cluster in enumerate(getattr(fusion_out, "disagreement_regions", [])):
                cb = cluster.canvas_box
                canvas_b = BoundingBox(x=cb[0], y=cb[1], width=cb[2], height=cb[3])
                unc_st = getattr(fusion_out, "uncertainty_state", "")
                findings_items.append(
                    FindingItem(
                        finding_id=f"f_disagree_{idx+1}",
                        type="cross_modal_disagreement",
                        summary=f"{cluster.label}: Divergent sensor signals ({cluster.notes}).",
                        bbox_px=cluster.pixel_box,
                        canvas_box=canvas_b,
                        geometry=cluster.geometry,
                        area_m2=cluster.area_sq_meters,
                        evidence_quality="WEAK" if unc_st == "CONFLICTING" else "SUPPORTED_WITH_WARNINGS",
                        source_model=getattr(fusion_out, "model_name", "SatQuery-OpticalSAR-FusionAdapter-v1"),
                        limitations=["Sensor divergence may indicate clouds, shadowing, vegetation canopy, or asphalt roughness"]
                    )
                )

        # 6. Semantic Safeguard Enforcement (Requirement 12)
        # If user asked an unsupported transition question like "Was forest converted to buildings?"
        is_unsupported_semantic = plan.classification_meta.get("unsupported_semantic_query", False)
        if is_unsupported_semantic:
            safeguard_prefix = (
                "Change was detected in these regions, but the active model cannot reliably "
                "determine the land-cover transition type."
            )
            answer_parts.insert(0, safeguard_prefix)
            warnings.append(
                "Semantic transition queries ('convert from X to Y') exceed verified binary change model scope."
            )

        # Default fallback synthesis if no answer was generated
        if not answer_parts:
            final_answer = "Analysis completed without generating structured textual findings."
        else:
            final_answer = " ".join(answer_parts)

        # Determine evidence_state
        evidence_state = "SUPPORTED"
        status = AnalysisStatus.COMPLETED
        if assessment:
            if assessment.evidence_status == "INSUFFICIENT":
                status = AnalysisStatus.INSUFFICIENT_EVIDENCE
                evidence_state = "INSUFFICIENT_EVIDENCE"
            elif assessment.evidence_status == "WEAK":
                status = AnalysisStatus.COMPLETED_WITH_WARNINGS
                evidence_state = "WEAK"
            elif assessment.evidence_status == "CONFLICTING":
                status = AnalysisStatus.COMPLETED_WITH_WARNINGS
                evidence_state = "CONFLICTING"
            elif any("DEGENERATE" in f or "MICRO" in f for f in assessment.quality_flags):
                status = AnalysisStatus.COMPLETED_WITH_WARNINGS
                evidence_state = "SUPPORTED_WITH_WARNINGS"

        if is_unsupported_semantic and evidence_state == "SUPPORTED":
            evidence_state = "SUPPORTED_WITH_WARNINGS"

        # Check out-of-domain flag
        has_out_of_domain = any("OUT_OF_DOMAIN" in w or "unsuitable" in w.lower() or "out of domain" in w.lower() or "too small" in w.lower() for w in warnings)
        if has_out_of_domain:
            evidence_state = "OUT_OF_DOMAIN"
            status = AnalysisStatus.INSUFFICIENT_EVIDENCE

        # 6. Overlays Data
        overlays: Optional[OverlayData] = None
        if temporal_obj:
            t1_url = f"/api/uploads/previews/{assets[0].id}.png" if len(assets) > 0 else None
            t2_url = f"/api/uploads/previews/{assets[1].id}.png" if len(assets) > 1 else None
            overlays = OverlayData(
                change_mask_url=getattr(temporal_obj, "mask_preview_url", None),
                t1_preview_url=t1_url,
                t2_preview_url=t2_url,
                change_ratio_pct=getattr(temporal_obj, "change_ratio_pct", None),
                total_changed_pixels=getattr(temporal_obj, "total_changed_pixels", None),
                georeferenced=getattr(temporal_obj, "has_authentic_geo", False)
            )
        elif fusion_out:
            t1_url = f"/api/uploads/previews/{assets[0].id}.png" if len(assets) > 0 else None
            t2_url = f"/api/uploads/previews/{assets[1].id}.png" if len(assets) > 1 else None
            overlays = OverlayData(
                change_mask_url=None,
                t1_preview_url=t1_url,
                t2_preview_url=t2_url,
                change_ratio_pct=getattr(fusion_out, "agreement_ratio_pct", None),
                total_changed_pixels=None,
                georeferenced=bool(assets[0].crs and len(assets) > 1 and assets[1].crs)
            )

        # 7. Registration Data
        reg_info: Optional[RegistrationInfo] = None
        if temporal_obj and getattr(temporal_obj, "registration_assessment", None):
            ra = temporal_obj.registration_assessment
            reg_info = RegistrationInfo(
                status=ra.get("quality_status", "ACCEPTABLE"),
                correlation=float(ra.get("normalized_correlation", 1.0)),
                translation_px=float(ra.get("translation_magnitude_px", 0.0)),
                warnings=ra.get("warnings", [])
            )
        elif fusion_out and getattr(fusion_out, "registration_assessment", None):
            fra = fusion_out.registration_assessment
            reg_info = RegistrationInfo(
                status=fra.quality_status,
                correlation=fra.normalized_mutual_information,
                translation_px=0.0,
                warnings=fra.warnings
            )

        # 8. Domain Suitability Data
        domain_info = DomainSuitabilityInfo(
            is_suitable=not has_out_of_domain,
            status="SUPPORTED" if not has_out_of_domain else "OUT_OF_DOMAIN",
            reasons=["Optical RGB true-color imagery within resolution and dimension specifications."] if not has_out_of_domain else ["Input violates operational domain requirements."],
            warnings=[w for w in warnings if "domain" in w.lower() or "unsuitable" in w.lower()]
        )

        # 9. Model Provenance Data
        is_temporal_plan = plan.task_family in [TaskFamily.TEMPORAL_CHANGE, TaskFamily.TEMPORAL_CHANGE_VQA]
        is_fusion_plan = plan.task_family == TaskFamily.OPTICAL_SAR_ANALYSIS
        if is_temporal_plan:
            prov_info = ModelProvenanceInfo(
                model_name="AttentionChangeNet",
                architecture="Dual ResNet-18 Siamese Backbone + Spatial Difference Attention",
                checkpoint="models/satquery_change_v1/attention_best.safetensors",
                parameters=12562474,
                threshold=0.50,
                benchmark_f1=0.8459,
                benchmark_iou=0.7330
            )
        elif is_fusion_plan:
            prov_info = ModelProvenanceInfo(
                model_name="SatQuery-OpticalSAR-FusionAdapter-v1",
                architecture="Cross-Modal Evidence Fusion + Enhanced Lee Speckle Filter + Specular Backscatter Gate",
                checkpoint="analytic/calibrated_backscatter_v1",
                parameters=0,
                threshold=0.0,
                benchmark_f1=0.8820,
                benchmark_iou=0.7890
            )
        else:
            prov_info = ModelProvenanceInfo(
                model_name="google/owlvit-base-patch32",
                architecture="Vision-Transformer OWL-ViT",
                checkpoint="google/owlvit-base-patch32",
                parameters=150000000,
                threshold=0.10,
                benchmark_f1=0.4500,
                benchmark_iou=0.3500
            )

        # 10. Downloadable Artifacts
        artifacts = DownloadableArtifacts(
            report_pdf_url=f"/api/analysis/{analysis_id}/report.pdf",
            result_json_url=f"/api/analysis/{analysis_id}/export.json",
            mask_png_url=overlays.change_mask_url if overlays else None,
            geojson_url=f"/api/analysis/{analysis_id}/export.geojson" if (overlays and overlays.georeferenced) else None
        )

        return AnalysisResult(
            id=analysis_id,
            answer=final_answer,
            task=plan.task_family,
            confidence=None,
            evidence=evidence_items,
            findings=findings_items,
            warnings=list(dict.fromkeys(warnings)),
            execution_summary=execution_trace,
            status=status,
            evidence_assessment=assessment,
            overlays=overlays,
            registration=reg_info,
            domain_suitability=domain_info,
            model_provenance=prov_info,
            downloadable_artifacts=artifacts,
            evidence_state=evidence_state,
            limitations=list(dict.fromkeys(warnings + getattr(fusion_out, "limitations", [])))
        )
