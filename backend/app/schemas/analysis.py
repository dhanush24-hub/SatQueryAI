from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.schemas.evidence import EvidenceItem, BoundingBox
from app.schemas.execution import ExecutionStep


class AnalysisRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language question about the imagery")
    image_ids: List[str] = Field(..., min_length=1, max_length=6, description="List of 1-6 uploaded ImageAsset IDs")
    session_id: Optional[str] = Field(None, description="Optional mission or conversation session ID")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional execution parameters")


class FindingItem(BaseModel):
    finding_id: str = Field(..., description="Unique finding identifier (e.g. finding_1)")
    type: str = Field(default="observed_change", description="Type of finding: observed_change, single_grounding, or vqa_observation")
    summary: str = Field(..., description="Concise finding summary description")
    bbox_px: Optional[List[int]] = Field(None, description="Bounding box in pixel coordinates [xmin, ymin, xmax, ymax]")
    canvas_box: Optional[BoundingBox] = Field(None, description="Normalized bounding box percentages [x, y, w, h]")
    geometry: Optional[Dict[str, Any]] = Field(None, description="GeoJSON polygon geometry (strictly when authentic CRS exists)")
    changed_pixels: Optional[int] = Field(None, description="Pixel count of change within this finding")
    area_m2: Optional[float] = Field(None, description="Physical area in m² (null if unreferenced imagery)")
    evidence_quality: str = Field(default="SUPPORTED", description="SUPPORTED, SUPPORTED_WITH_WARNINGS, WEAK, INSUFFICIENT_EVIDENCE, OUT_OF_DOMAIN, UNAVAILABLE")
    source_model: str = Field(default="AttentionChangeNet", description="Name/checkpoint of the generating model")
    limitations: List[str] = Field(default_factory=list, description="Specific limitations or caveats for this finding")


class OverlayData(BaseModel):
    change_mask_url: Optional[str] = Field(None, description="URL to PNG change mask preview")
    t1_preview_url: Optional[str] = Field(None, description="URL to T1 image preview")
    t2_preview_url: Optional[str] = Field(None, description="URL to T2 image preview")
    change_ratio_pct: Optional[float] = Field(None, description="Percentage of valid scene flagged as changed")
    total_changed_pixels: Optional[int] = Field(None, description="Total count of positive change pixels")
    georeferenced: bool = Field(default=False, description="Whether authentic georeferencing metadata exists")


class RegistrationInfo(BaseModel):
    status: str = Field(default="UNCHECKED", description="ACCEPTABLE, MARGINAL, UNACCEPTABLE, or UNCHECKED")
    correlation: float = Field(default=1.0, description="Normalized mutual correlation score [0-1]")
    translation_px: float = Field(default=0.0, description="Estimated translation magnitude in pixels")
    warnings: List[str] = Field(default_factory=list, description="Registration alignment warnings")


class DomainSuitabilityInfo(BaseModel):
    is_suitable: bool = Field(default=True, description="Whether imagery is within model's supported operational domain")
    status: str = Field(default="SUPPORTED", description="SUPPORTED, SUPPORTED_WITH_WARNINGS, or OUT_OF_DOMAIN")
    reasons: List[str] = Field(default_factory=list, description="Justification reasons for domain decision")
    warnings: List[str] = Field(default_factory=list, description="Domain-specific cautions")


class ModelProvenanceInfo(BaseModel):
    model_name: str = Field(default="AttentionChangeNet", description="Official model name")
    architecture: str = Field(default="Dual ResNet-18 Siamese Backbone + Spatial Difference Attention", description="Architecture specification")
    checkpoint: str = Field(default="models/satquery_change_v1/attention_best.safetensors", description="Checkpoint file path")
    parameters: int = Field(default=12562474, description="Parameter count")
    threshold: float = Field(default=0.50, description="Calibrated decision threshold")
    benchmark_f1: float = Field(default=0.8459, description="Validated Global F1 on LEVIR-CD held-out test split")
    benchmark_iou: float = Field(default=0.7330, description="Validated Global IoU on LEVIR-CD held-out test split")


class DownloadableArtifacts(BaseModel):
    report_pdf_url: str = Field(..., description="Download URL for PDF analysis report")
    result_json_url: str = Field(..., description="Download URL for full result JSON")
    mask_png_url: Optional[str] = Field(None, description="Download URL for change mask PNG")
    geojson_url: Optional[str] = Field(None, description="Download URL for authentic GeoJSON (null if unreferenced)")


class AnalysisHistorySummary(BaseModel):
    id: str = Field(..., description="Analysis ID")
    query: str = Field(..., description="User query")
    timestamp: str = Field(..., description="Execution timestamp (ISO)")
    image_ids: List[str] = Field(default_factory=list, description="Input image IDs")
    task: Optional[TaskFamily] = Field(None, description="Task family executed")
    status: AnalysisStatus = Field(default=AnalysisStatus.COMPLETED, description="Status")
    thumbnail: Optional[str] = Field(None, description="Thumbnail URL")
    summary: str = Field(..., description="Short textual summary of result")
    evidence_state: str = Field(default="SUPPORTED", description="Evidence quality state")


class AnalysisResult(BaseModel):
    id: str = Field(..., description="Unique analysis result ID")
    answer: str = Field(..., description="Synthesized geospatial answer")
    task: Optional[TaskFamily] = Field(default=TaskFamily.UNKNOWN, description="Identified task family executed")
    confidence: Optional[str] = Field(None, description="Grounded confidence statement (calibrated or qualitative)")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Validated spatial and temporal evidence items")
    warnings: List[str] = Field(default_factory=list, description="Warnings or limitations encountered during analysis")
    execution_summary: List[ExecutionStep] = Field(default_factory=list, description="Observable step-by-step execution trace")
    status: AnalysisStatus = Field(default=AnalysisStatus.COMPLETED, description="Overall status of the analysis")
    plan_id: Optional[str] = Field(None, description="Executed workflow plan identifier")
    evidence_assessment: Optional[Any] = Field(None, description="Detailed evidence quality and verification assessment")

    # Prompt 7 Evidence-First Extensions
    findings: List[FindingItem] = Field(default_factory=list, description="Structured regional and semantic findings")
    overlays: Optional[OverlayData] = Field(None, description="Imagery overlays and change mask metadata")
    registration: Optional[RegistrationInfo] = Field(None, description="Coregistration quality assessment")
    domain_suitability: Optional[DomainSuitabilityInfo] = Field(None, description="Operational domain compliance check")
    model_provenance: Optional[ModelProvenanceInfo] = Field(None, description="Traceable model architecture and checkpoint provenance")
    downloadable_artifacts: Optional[DownloadableArtifacts] = Field(None, description="Export URLs for PDF, JSON, Mask, and GeoJSON")
    evidence_state: str = Field(default="SUPPORTED", description="Overall evidence quality state: SUPPORTED, SUPPORTED_WITH_WARNINGS, WEAK, INSUFFICIENT_EVIDENCE, OUT_OF_DOMAIN, UNAVAILABLE")
    limitations: List[str] = Field(default_factory=list, description="Aggregated operational and sensor limitations")
