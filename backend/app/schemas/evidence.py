from typing import List, Optional
from pydantic import BaseModel, Field
from app.domain.tasks import EvidenceVerdict


class BoundingBox(BaseModel):
    x: float = Field(..., ge=0.0, le=100.0, description="Normalized X origin percentage [0-100]")
    y: float = Field(..., ge=0.0, le=100.0, description="Normalized Y origin percentage [0-100]")
    width: float = Field(..., gt=0.0, le=100.0, description="Normalized width percentage [0-100]")
    height: float = Field(..., gt=0.0, le=100.0, description="Normalized height percentage [0-100]")


class EvidenceItem(BaseModel):
    id: str = Field(..., description="Unique evidence identifier (e.g. e1, e2)")
    image_id: str = Field(..., description="Referenced ImageAsset ID")
    finding_id: str = Field(..., description="Associated finding ID")
    evidence_type: str = Field(..., description="visual_region, temporal_difference, or cross_sensor_support")
    description: str = Field(..., description="Descriptive spatial context for this evidence")
    region: Optional[BoundingBox] = Field(None, description="Spatial bounding box ROI")
    reliability: str = Field(default="medium", description="Evidence reliability assessment (high, medium, low)")
    limitations: List[str] = Field(default_factory=list, description="Sensor or spatial limitations")
    source_step: str = Field(default="unknown", description="Tool or step that produced this evidence item")


class EvidenceValidation(BaseModel):
    finding_id: str = Field(..., description="Target finding ID")
    verdict: EvidenceVerdict = Field(..., description="Deterministic evidence gate verdict")
    reasons: List[str] = Field(default_factory=list, description="Reasons justifying the verdict")
    recommendation: Optional[str] = Field(None, description="Guidance if evidence is insufficient or conflicting")
