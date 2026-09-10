"""
Input domain validation and suitability gate for remote sensing intelligence tasks.
Ensures models are only evaluated on supported distributions and tasks fail cleanly
with clear explanations when given out-of-domain or degraded imagery.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DomainSuitability(str, Enum):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_WARNINGS = "SUPPORTED_WITH_WARNINGS"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class DomainAssessment(BaseModel):
    suitability: DomainSuitability
    is_suitable: bool
    reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    supported_tasks: List[str] = Field(default_factory=list)
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)


def assess_input_domain(
    modality: Optional[str],
    num_bands: int,
    dimensions: tuple,  # (H, W)
    resolution_m: Optional[float] = None,
    registration_status: Optional[str] = None,
    cloud_or_nodata_ratio: float = 0.0,
    dynamic_range: Optional[tuple] = None,  # (min_val, max_val)
    task: str = "temporal_change"
) -> DomainAssessment:
    """
    Evaluates whether an input image or temporal pair satisfies the requirements
    of the remote-sensing intelligence pipeline.
    """
    reasons: List[str] = []
    warnings: List[str] = []
    h, w = dimensions

    # 1. Modality check
    norm_mod = (modality or "").upper()
    if norm_mod in ["SAR", "RADAR"]:
        return DomainAssessment(
            suitability=DomainSuitability.OUT_OF_DOMAIN,
            is_suitable=False,
            reasons=[f"Modality '{modality}' is not supported by optical change detection."],
            warnings=["SAR imagery requires specialized speckle filtering and radar-specific change metrics."],
            supported_tasks=["metadata_inspection"]
        )

    # 2. Dimensions check
    if h < 64 or w < 64:
        return DomainAssessment(
            suitability=DomainSuitability.INSUFFICIENT_EVIDENCE,
            is_suitable=False,
            reasons=[f"Spatial dimensions {w}x{h} are too small (minimum required is 64x64)."],
            warnings=["Sub-patch resolution lacks sufficient context for change localization."],
            supported_tasks=[]
        )

    # 3. Registration check (for temporal tasks)
    if task in ["temporal_change", "temporal_vqa"]:
        if registration_status == "UNACCEPTABLE":
            return DomainAssessment(
                suitability=DomainSuitability.INSUFFICIENT_EVIDENCE,
                is_suitable=False,
                reasons=["Geometric registration error between T1 and T2 exceeds acceptable tolerance."],
                warnings=["Co-registration failure would generate massive false-positive edge changes."],
                supported_tasks=["co_registration_alignment"]
            )
        elif registration_status == "MARGINAL":
            warnings.append("Registration is marginal; fine structural changes may contain alignment artifacts.")

    # 4. Nodata/cloud ratio
    if cloud_or_nodata_ratio > 0.60:
        return DomainAssessment(
            suitability=DomainSuitability.INSUFFICIENT_EVIDENCE,
            is_suitable=False,
            reasons=[f"Nodata or cloud occlusion ratio ({cloud_or_nodata_ratio*100:.1f}%) exceeds 60% threshold."],
            warnings=["Insufficient clear-sky terrestrial pixels available for change assessment."],
            supported_tasks=[]
        )
    elif cloud_or_nodata_ratio > 0.20:
        warnings.append(f"Significant nodata or occlusion ({cloud_or_nodata_ratio*100:.1f}%) observed.")

    # 5. Band count
    if num_bands < 3:
        warnings.append(f"Single-band or panchromatic imagery ({num_bands} band) replicated to 3 channels.")

    # 6. Resolution / GSD check
    if resolution_m is not None:
        if resolution_m > 30.0:
            warnings.append(f"Coarse spatial resolution ({resolution_m:.1f}m GSD) may obscure building-scale changes.")
        elif resolution_m < 0.05:
            warnings.append(f"Ultra-high resolution ({resolution_m:.2f}m GSD) exceeds typical satellite training domain.")

    suitability = DomainSuitability.SUPPORTED_WITH_WARNINGS if warnings else DomainSuitability.SUPPORTED

    return DomainAssessment(
        suitability=suitability,
        is_suitable=True,
        reasons=reasons,
        warnings=warnings,
        supported_tasks=["temporal_change", "temporal_vqa", "feature_grounding"],
        metrics_summary={
            "dimensions": f"{w}x{h}",
            "bands": num_bands,
            "resolution_m": resolution_m,
            "cloud_or_nodata_ratio": round(cloud_or_nodata_ratio, 4)
        }
    )
