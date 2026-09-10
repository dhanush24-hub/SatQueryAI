from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class PairType(str, Enum):
    TEMPORAL = "TEMPORAL"
    OPTICAL_SAR = "OPTICAL_SAR"


class PairCompatibilityRequest(BaseModel):
    asset_id_a: str = Field(..., description="First ImageAsset ID (e.g. Baseline T1 or Optical)")
    asset_id_b: str = Field(..., description="Second ImageAsset ID (e.g. Target T2 or SAR)")
    pair_type: PairType = Field(default=PairType.TEMPORAL, description="Type of pair analysis requested")


class PairCompatibilityReport(BaseModel):
    compatible: bool = Field(..., description="Whether the pair can be analyzed together")
    pair_type: PairType = Field(..., description="Pair type requested")
    asset_id_a: str
    asset_id_b: str
    crs_a: Optional[str] = None
    crs_b: Optional[str] = None
    crs_match: bool = Field(default=False, description="Whether both images share the exact same CRS")
    spatial_overlap_percentage: float = Field(default=0.0, description="Estimated geographic overlap percentage [0-100%]")
    has_spatial_overlap: bool = Field(default=False, description="Whether footprints intersect geographically")
    overlap_geographic_bbox: Optional[List[float]] = Field(None, description="WGS84 bounding box of intersection [minLon, minLat, maxLon, maxLat]")
    resolution_ratio: float = Field(default=1.0, description="Ratio of resolutions (GSD_A / GSD_B)")
    requires_reprojection: bool = Field(default=False, description="Whether CRS reprojection is needed")
    requires_resampling: bool = Field(default=False, description="Whether pixel grid resampling is needed")
    is_coregistered: bool = Field(default=False, description="Whether identical grid and pixel co-registration is verified")
    modality_compatible: bool = Field(default=True, description="Whether modalities match the requested pair type")
    acquisition_dates: Optional[List[Optional[str]]] = None
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AlignmentRequest(BaseModel):
    source_asset_id: str = Field(..., description="Asset to be reprojected/resampled")
    reference_asset_id: str = Field(..., description="Asset whose grid and CRS define the target alignment")
    resampling_method: str = Field(default="bilinear", description="Resampling algorithm: nearest, bilinear, cubic")
