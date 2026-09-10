from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.domain.modalities import Modality, ImageFormat


class ImageAsset(BaseModel):
    id: str = Field(..., description="Unique identifier for the image asset")
    filename: str = Field(..., description="Unique filename in storage")
    storage_path: str = Field(..., description="Sanitized storage key (no absolute server paths exposed)")
    original_filename: str = Field(..., description="Name of the file as uploaded by user")
    format: ImageFormat = Field(default=ImageFormat.UNKNOWN, description="Image raster format")
    modality: Modality = Field(default=Modality.UNKNOWN, description="Sensor modality")
    width: Optional[int] = Field(None, description="Raster width in pixels")
    height: Optional[int] = Field(None, description="Raster height in pixels")
    bands: Optional[int] = Field(None, description="Number of spectral/polarization bands")
    crs: Optional[str] = Field(None, description="Coordinate Reference System string (e.g. EPSG:4326)")
    resolution: Optional[float] = Field(None, description="Ground Sample Distance in meters")
    bbox: Optional[List[float]] = Field(None, description="Projected bounding box [minX, minY, maxX, maxY]")
    geographic_bbox: Optional[List[float]] = Field(None, description="WGS84 geographic bounding box [minLon, minLat, maxLon, maxLat]")
    nodata: Optional[float] = Field(None, description="NoData pixel value")
    acquisition_time: Optional[str] = Field(None, description="ISO-8601 acquisition timestamp if known from metadata")
    sensor: Optional[str] = Field(None, description="Sensor platform / instrument if verified from metadata tags")
    affine_transform: Optional[List[float]] = Field(None, description="Affine geotransform matrix (6 or 9 elements)")
    driver: Optional[str] = Field(None, description="GDAL driver name (e.g. GTiff)")
    band_descriptions: List[str] = Field(default_factory=list, description="Descriptions of each band from tags")
    dtypes: List[str] = Field(default_factory=list, description="Data types of each band")
    preview_url: Optional[str] = Field(None, description="URL to retrieve the generated browser preview image")
    has_preview: bool = Field(default=False, description="Whether a valid display preview was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw geospatial and raster metadata tags")
    validation_status: str = Field(default="valid", description="Asset validation status (valid, partial, invalid)")
    warnings: List[str] = Field(default_factory=list, description="Validation or extraction warnings")
