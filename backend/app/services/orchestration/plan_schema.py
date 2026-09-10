from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.domain.tasks import TaskFamily


class PlanStep(BaseModel):
    step_id: str = Field(..., description="Unique step identifier in the plan DAG")
    tool_id: str = Field(..., description="Registered ModelToolRegistry tool identifier")
    task_family: TaskFamily = Field(..., description="Task family executed by this step")
    description: str = Field(..., description="Human-readable description of this step")
    input_bindings: Dict[str, Any] = Field(default_factory=dict, description="Bindings to inputs or prior step outputs")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Permitted execution parameters")
    dependencies: List[str] = Field(default_factory=list, description="List of prerequisite step_ids")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0, description="Execution timeout limit")
    status: str = Field(default="pending", description="Step lifecycle status")


class WorkflowPlan(BaseModel):
    plan_id: str = Field(..., description="Unique identifier for this workflow plan")
    task_family: TaskFamily = Field(..., description="Overall classified target task family")
    input_asset_ids: List[str] = Field(..., description="Asset IDs required for this plan")
    steps: List[PlanStep] = Field(..., description="Ordered/DAG sequence of execution steps")
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Step dependency mapping")
    permitted_parameters: Dict[str, Any] = Field(default_factory=dict, description="Allowlisted parameter bounds")
    expected_outputs: List[str] = Field(default_factory=list, description="Expected output artifact names")
    classification_meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata from task classification")
    validation_status: str = Field(default="PENDING", description="Validation status: VALID | INVALID | BLOCKED")
    rejection_reason: Optional[str] = Field(None, description="Detailed reason if rejected or blocked")


class EvidenceAssessment(BaseModel):
    evidence_status: str = Field(
        ...,
        description="Assessment status: SUPPORTED | WEAK | CONFLICTING | INSUFFICIENT | UNAVAILABLE"
    )
    raw_scores: List[float] = Field(default_factory=list, description="Preserved raw model scores without calibration")
    quality_flags: List[str] = Field(default_factory=list, description="Quality and integrity warning flags")
    limitations: List[str] = Field(default_factory=list, description="Known model, dataset, or heuristic limitations")
    verification_method: str = Field(default="uncalibrated_heuristic", description="Method used to assess evidence")
    confidence: Optional[str] = Field(None, description="Calibrated confidence statement, strictly null if uncalibrated")
    bounding_box_count: int = Field(default=0, description="Number of visual regions assessed")
    geometric_anomalies: List[str] = Field(default_factory=list, description="Geometric anomaly warnings")
