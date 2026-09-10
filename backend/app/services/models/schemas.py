from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.domain.modalities import Modality
from app.domain.tasks import TaskFamily


class VqaRequest(BaseModel):
    asset_id: str = Field(..., description="ImageAsset ID to analyze")
    question: str = Field(..., description="Natural language visual question")
    max_tokens: int = Field(default=64, description="Maximum tokens to decode")


class VqaResponse(BaseModel):
    answer: str
    model_name: str
    task_family: TaskFamily = TaskFamily.SINGLE_VQA
    device: str
    inference_latency_ms: float
    confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroundingBox(BaseModel):
    label: str
    score: float
    # Normalized coordinates [0.0 - 100.0%] for frontend canvas rendering
    canvas_box: List[float] = Field(..., description="[x_min_pct, y_min_pct, width_pct, height_pct]")
    # Pixel coordinates [x_min, y_min, x_max, y_max] in original raster space
    pixel_box: List[int] = Field(..., description="[xmin, ymin, xmax, ymax]")
    # Geographic coordinates [minLon, minLat, maxLon, maxLat] if georeferenced
    geographic_bbox: Optional[List[float]] = Field(None, description="[minLon, minLat, maxLon, maxLat]")


class GroundingRequest(BaseModel):
    asset_id: str = Field(..., description="ImageAsset ID to ground features in")
    queries: List[str] = Field(..., description="List of referring expressions or object labels to locate")
    threshold: float = Field(default=0.10, description="Minimum confidence threshold for detected regions")


class GroundingResponse(BaseModel):
    boxes: List[GroundingBox]
    model_name: str
    task_family: TaskFamily = TaskFamily.SINGLE_GROUNDING
    device: str
    inference_latency_ms: float
    queries: List[str]
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Temporal Change Schemas
# =========================================================================

class TemporalChangeCluster(BaseModel):
    cluster_id: str
    label: str = "detected_change"
    score: float
    canvas_box: List[float] = Field(..., description="[x_min_pct, y_min_pct, width_pct, height_pct]")
    pixel_box: List[int] = Field(..., description="[xmin, ymin, xmax, ymax]")
    geographic_bbox: Optional[List[float]] = None
    geometry: Optional[Dict[str, Any]] = None
    area_sq_meters: Optional[float] = None
    area_hectares: Optional[float] = None
    pixel_count: int = 0
    centroid_geo: Optional[List[float]] = None


class TemporalChangeRequest(BaseModel):
    t1_asset_id: str = Field(..., description="Pre-change ImageAsset ID")
    t2_asset_id: str = Field(..., description="Post-change ImageAsset ID")
    threshold: float = Field(default=0.50, description="Threshold on change probability")
    method: str = Field(default="siamunet_diff", description="'siamunet_diff' or 'analytical_cva'")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TemporalChangeResponse(BaseModel):
    task_family: TaskFamily = TaskFamily.TEMPORAL_CHANGE
    model_name: str
    device: str
    inference_latency_ms: float
    change_clusters: List[TemporalChangeCluster] = Field(default_factory=list)
    total_changed_pixels: int = 0
    total_valid_pixels: int = 0
    change_ratio_pct: float = 0.0
    changed_area_m2: Optional[float] = None
    changed_area_ha: Optional[float] = None
    changed_area_sq_km: Optional[float] = None
    has_authentic_geo: bool = False
    change_verdict: str = "NO_CHANGE_DETECTED"  # OBSERVED_CHANGE, POSSIBLE_CHANGE, NO_CHANGE_DETECTED, INSUFFICIENT_EVIDENCE
    mask_preview_url: Optional[str] = None
    registration_assessment: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemporalVqaRequest(BaseModel):
    t1_asset_id: str = Field(..., description="Pre-change ImageAsset ID")
    t2_asset_id: str = Field(..., description="Post-change ImageAsset ID")
    question: str = Field(..., description="Natural language question about temporal changes")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TemporalVqaResponse(BaseModel):
    task_family: TaskFamily = TaskFamily.TEMPORAL_CHANGE_VQA
    model_name: str
    device: str
    inference_latency_ms: float
    answer: str
    change_verdict: str = "NO_CHANGE_DETECTED"
    change_clusters: List[TemporalChangeCluster] = Field(default_factory=list)
    changed_area_ha: Optional[float] = None
    change_ratio_pct: float = 0.0
    total_changed_pixels: int = 0
    has_authentic_geo: bool = False
    mask_preview_url: Optional[str] = None
    registration_assessment: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
