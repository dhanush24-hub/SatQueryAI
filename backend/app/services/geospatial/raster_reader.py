import os
from typing import Any, Dict, List, Optional, Tuple
import rasterio
from rasterio.warp import transform_bounds
from rasterio.errors import RasterioIOError, CRSError
from app.domain.modalities import Modality, ImageFormat
from app.core.logging import logger
from app.services.geospatial.validation import MAX_RASTER_DIMENSION, MAX_BAND_COUNT


class RasterMetadata:
    def __init__(
        self,
        width: int,
        height: int,
        bands: int,
        dtypes: List[str],
        driver: str,
        crs: Optional[str],
        resolution: Optional[float],
        bbox: Optional[List[float]],
        geographic_bbox: Optional[List[float]],
        affine_transform: List[float],
        nodata: Optional[float],
        band_descriptions: List[str],
        color_interp: List[str],
        tags: Dict[str, Any],
        acquisition_time: Optional[str],
        sensor: Optional[str],
        modality: Modality,
        warnings: List[str],
    ):
        self.width = width
        self.height = height
        self.bands = bands
        self.dtypes = dtypes
        self.driver = driver
        self.crs = crs
        self.resolution = resolution
        self.bbox = bbox
        self.geographic_bbox = geographic_bbox
        self.affine_transform = affine_transform
        self.nodata = nodata
        self.band_descriptions = band_descriptions
        self.color_interp = color_interp
        self.tags = tags
        self.acquisition_time = acquisition_time
        self.sensor = sensor
        self.modality = modality
        self.warnings = warnings


def inspect_raster(filepath: str, user_modality: Optional[Modality] = None) -> RasterMetadata:
    """
    Inspect an ingested raster using Rasterio.
    Extracts real geospatial metadata without guessing or fabricating data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError("Target raster file does not exist in storage.")

    try:
        with rasterio.open(filepath) as src:
            width = src.width
            height = src.height
            bands = src.count

            # Dimension limits
            if width > MAX_RASTER_DIMENSION or height > MAX_RASTER_DIMENSION:
                raise ValueError(
                    f"Raster dimensions ({width}x{height}) exceed maximum allowed limit of {MAX_RASTER_DIMENSION}px."
                )

            if bands > MAX_BAND_COUNT or bands < 1:
                raise ValueError(f"Raster band count ({bands}) is out of allowed range [1-{MAX_BAND_COUNT}].")

            dtypes = [str(dt) for dt in src.dtypes]
            driver = src.driver
            nodata = float(src.nodata) if src.nodata is not None else None

            # Affine transform
            t = src.transform
            affine_transform = [t.a, t.b, t.c, t.d, t.e, t.f]

            # Resolution (GSD)
            res_x, res_y = src.res
            resolution = round(float((abs(res_x) + abs(res_y)) / 2.0), 4)

            # CRS extraction
            crs_str = None
            if src.crs:
                if src.crs.is_epsg_code:
                    crs_str = f"EPSG:{src.crs.to_epsg()}"
                else:
                    crs_str = src.crs.to_string()

            # Bounding box in projected coordinates
            b = src.bounds
            bbox = [round(b.left, 6), round(b.bottom, 6), round(b.right, 6), round(b.top, 6)]

            # Geographic bounding box in WGS84 (EPSG:4326)
            geographic_bbox = None
            warnings: List[str] = []

            if src.crs:
                try:
                    if src.crs.to_epsg() == 4326:
                        geographic_bbox = bbox
                    else:
                        gb = transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
                        geographic_bbox = [round(gb[0], 6), round(gb[1], 6), round(gb[2], 6), round(gb[3], 6)]
                except Exception as e:
                    warnings.append(f"Failed to transform bounds to WGS84: {str(e)}")
            else:
                warnings.append("No Coordinate Reference System (CRS) found; raster is unreferenced pixel space.")

            # Tags and metadata
            tags = src.tags()
            band_descriptions = [src.descriptions[i] or f"Band_{i+1}" for i in range(bands)]
            color_interp = [ci.name for ci in src.colorinterp]

            # Timestamp extraction from known EXIF/TIFF/GDAL tags (no guessing)
            acquisition_time = None
            date_tag_candidates = ["TIFFTAG_DATETIME", "ACQUISITION_DATE", "DATETIME", "time", "date_time", "ACQUISITIONDATETIME"]
            for tag in date_tag_candidates:
                if tag in tags:
                    acquisition_time = str(tags[tag]).strip()
                    break

            # Sensor / Platform extraction from metadata tags (never from filename)
            sensor = None
            sensor_tag_candidates = ["SATELLITE", "SPACECRAFT_NAME", "MISSION", "SENSOR", "PLATFORM", "INSTRUMENT", "SENSOR_ID"]
            for tag in sensor_tag_candidates:
                if tag in tags:
                    sensor = str(tags[tag]).strip()
                    break

            # Modality determination (strictly metadata-driven)
            modality = Modality.UNKNOWN
            all_meta_str = " ".join([str(v) for v in tags.values()] + band_descriptions + color_interp).upper()

            # Check explicit user input first
            if user_modality and user_modality != Modality.UNKNOWN:
                modality = user_modality
            else:
                # Inspect for SAR signatures in band names or tags
                is_sar_tags = any(k in all_meta_str for k in ["SAR", "C-SAR", "RADAR", "BACKSCATTER", "SIGMA0", "GAMMA0"])
                has_sar_pols = any(p in all_meta_str for p in ["VV", "VH", "HH", "HV"])
                if is_sar_tags or has_sar_pols:
                    modality = Modality.SAR
                # Inspect for multispectral signatures
                elif bands > 3 or any(k in all_meta_str for k in ["NIR", "SWIR", "RE1", "RE2", "B08", "B11"]):
                    modality = Modality.MULTISPECTRAL
                # Inspect for true color / RGB optical
                elif bands == 3 and any(k in all_meta_str for k in ["RED", "GREEN", "BLUE"]):
                    modality = Modality.OPTICAL
                elif bands in [1, 3] and driver in ["PNG", "JPEG"]:
                    modality = Modality.OPTICAL
                else:
                    modality = Modality.UNKNOWN
                    warnings.append("Modality could not be verified from raster tags; marked as UNKNOWN.")

            return RasterMetadata(
                width=width,
                height=height,
                bands=bands,
                dtypes=dtypes,
                driver=driver,
                crs=crs_str,
                resolution=resolution,
                bbox=bbox,
                geographic_bbox=geographic_bbox,
                affine_transform=affine_transform,
                nodata=nodata,
                band_descriptions=band_descriptions,
                color_interp=color_interp,
                tags=tags,
                acquisition_time=acquisition_time,
                sensor=sensor,
                modality=modality,
                warnings=warnings,
            )

    except (RasterioIOError, CRSError) as e:
        logger.error(f"Failed to open or parse raster: {str(e)}")
        raise ValueError("The uploaded file is corrupt or not a readable geospatial raster.")
    except Exception as e:
        logger.error(f"Error during raster inspection: {str(e)}")
        raise ValueError(f"Raster inspection failed: {str(e)}")
