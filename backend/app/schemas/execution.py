from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ExecutionStep(BaseModel):
    """
    Exposes strictly observable execution telemetry for auditable workflows.
    Hidden chain-of-thought, internal prompt scratchpads, and reasoning tokens
    must never be exposed in this model.
    """
    step: str = Field(..., description="Human-readable name of the execution step")
    tool_name: str = Field(..., description="Identifier of the specialist tool invoked")
    model_name: Optional[str] = Field(None, description="Identifier of the remote sensing model checkpoint")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Observable operational arguments (e.g. thresholds, band indices)")
    status: str = Field(..., description="Execution status (pending, running, completed, failed, skipped)")
    duration_ms: Optional[int] = Field(None, description="Step duration in milliseconds")
    summary: str = Field(..., description="Observable factual output summary of the step")
