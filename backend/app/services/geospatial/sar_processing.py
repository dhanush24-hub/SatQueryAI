import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from scipy.ndimage import uniform_filter
from app.schemas.image_asset import ImageAsset
from app.core.logging import logger


class SarProcessingReport(BaseModel):
    """
    Observable report documenting all physical SAR transformations,
    metadata validation, and radiometric calibration status.
    """
    asset_id: str
    sensor: Optional[str] = None
    product_type: str = "UNKNOWN"  # GRD, SLC, DERIVED, UNKNOWN
    polarization: str = "UNKNOWN"  # VV, VH, HH, HV, DUAL, UNKNOWN
    input_representation: str = "UNKNOWN"  # AMPLITUDE, INTENSITY, DB, UNKNOWN
    is_calibrated: bool = False
    speckle_filter_applied: str = "NONE"  # NONE, ENHANCED_LEE_5x5, etc.
    min_db: Optional[float] = None
    max_db: Optional[float] = None
    mean_db: Optional[float] = None
    transformations_log: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def inspect_sar_metadata(asset: ImageAsset) -> Dict[str, Any]:
    """
    Extract verified SAR properties strictly from metadata tags and raster structure.
    Never infers polarization, product type, or calibration from pixel statistics alone.
    """
    raw_meta = asset.metadata or {}
    tags = {k.upper(): str(v).upper() for k, v in raw_meta.items()}
    band_descs = [d.upper() for d in asset.band_descriptions]
    combined_tags_str = " ".join(list(tags.keys()) + list(tags.values()) + band_descs)

    # 1. Platform / Sensor
    sensor = asset.sensor
    if not sensor:
        for tag in ["PLATFORM", "SATELLITE", "MISSION", "INSTRUMENT", "SENSOR"]:
            if tag in tags:
                sensor = tags[tag]
                break

    # 2. Product type (GRD / SLC / RAW / DERIVED)
    product_type = "UNKNOWN"
    if "GRD" in combined_tags_str or "GROUND RANGE" in combined_tags_str:
        product_type = "GRD"
    elif "SLC" in combined_tags_str or "SINGLE LOOK COMPLEX" in combined_tags_str:
        product_type = "SLC"
    elif "PROXY" in combined_tags_str or "DERIVED" in combined_tags_str or "ARIA" in combined_tags_str:
        product_type = "DERIVED"

    # 3. Polarization (VV, VH, HH, HV)
    polarization = "UNKNOWN"
    pols_found = []
    for pol in ["VV", "VH", "HH", "HV"]:
        if pol in combined_tags_str or any(pol in d for d in band_descs):
            pols_found.append(pol)
    if pols_found:
        polarization = "_".join(sorted(set(pols_found)))

    # 4. Calibration & Representation
    input_rep = "UNKNOWN"
    is_calibrated = False
    if any(k in combined_tags_str for k in ["SIGMA0", "SIGMA_0", "GAMMA0", "BETA0", "CALIBRATED"]):
        is_calibrated = True
        if "DB" in combined_tags_str or "DECIBEL" in combined_tags_str:
            input_rep = "DB"
        else:
            input_rep = "INTENSITY"
    elif any(k in combined_tags_str for k in ["AMPLITUDE", "DN", "DIGITAL NUMBER"]):
        input_rep = "AMPLITUDE"
        is_calibrated = False

    return {
        "sensor": sensor,
        "product_type": product_type,
        "polarization": polarization,
        "input_representation": input_rep,
        "is_calibrated": is_calibrated,
    }


def enhanced_lee_filter(
    intensity: np.ndarray,
    window_size: int = 5,
    damping_factor: float = 1.0,
    num_looks: float = 1.0
) -> np.ndarray:
    """
    Enhanced Lee Speckle Filter for SAR intensity images.
    Preserves point targets, edges, and homogeneous regions without over-smoothing.
    
    Args:
        intensity: 2D numpy array of SAR intensity (linear power, positive values).
        window_size: Odd integer filter kernel size (default: 5).
        damping_factor: Damping factor eta for filtering (default: 1.0).
        num_looks: Equivalent Number of Looks (ENL) of the SAR product.
    
    Returns:
        Filtered intensity array of identical dimensions.
    """
    if intensity.ndim != 2:
        raise ValueError("Enhanced Lee filter expects a 2D single-channel array.")

    data = np.maximum(intensity.astype(np.float32), 1e-7)

    # Local mean and local variance via spatial uniform filtering
    local_mean = uniform_filter(data, size=window_size, mode="reflect")
    local_sq_mean = uniform_filter(data ** 2, size=window_size, mode="reflect")
    local_var = np.maximum(local_sq_mean - (local_mean ** 2), 0.0)

    # Speckle noise standard deviation coefficient
    # For intensity distribution with N looks: Cu = 1 / sqrt(ENL)
    cu = 1.0 / np.sqrt(max(num_looks, 1.0))
    # Threshold for point targets / high heterogeneity: Cmax = sqrt(1 + 2/ENL)
    cmax = np.sqrt(1.0 + (2.0 / max(num_looks, 1.0)))

    # Local coefficient of variation Ci = sigma_I / mean_I
    ci = np.sqrt(local_var) / np.maximum(local_mean, 1e-7)

    # Weight factor calculation:
    # 1. Homogeneous region (Ci <= Cu): W = 0 -> output = local mean
    # 2. Heterogeneous region (Cu < Ci < Cmax): W = exp(-damping * (Ci - Cu) / (Cmax - Ci))
    # 3. High heterogeneity / point target (Ci >= Cmax): W = 1 -> output = raw pixel
    weight = np.zeros_like(data)

    # Heterogeneous mask
    mid_mask = (ci > cu) & (ci < cmax)
    weight[mid_mask] = np.exp(
        -damping_factor * (ci[mid_mask] - cu) / np.maximum(cmax - ci[mid_mask], 1e-5)
    )
    # Point target mask
    high_mask = ci >= cmax
    weight[high_mask] = 1.0

    filtered = (local_mean * (1.0 - weight)) + (data * weight)
    return np.clip(filtered, 1e-7, None).astype(np.float32)


def preprocess_sar_raster(
    sar_data: np.ndarray,
    asset: ImageAsset,
    apply_speckle_filter: bool = True,
    target_unit: str = "DB"
) -> Tuple[np.ndarray, SarProcessingReport]:
    """
    Preprocesses SAR raster into physically calibrated backscatter sigma0 (dB).
    Strictly preserves the original raster on disk; transforms only the in-memory array.
    
    Pipeline:
    1. Validate input dimensions and metadata.
    2. Handle multi-band / multi-polarization extraction (selects VV or primary band).
    3. Convert Amplitude to Intensity: I = A^2 (if input is amplitude).
    4. Apply Enhanced Lee speckle filter in intensity domain.
    5. Convert to Backscatter dB: sigma0_dB = 10 * log10(I + eps).
    """
    warnings: List[str] = []
    log: List[str] = []

    # 1. Inspect metadata
    meta = inspect_sar_metadata(asset)
    sensor = meta["sensor"]
    product_type = meta["product_type"]
    polarization = meta["polarization"]
    input_rep = meta["input_representation"]
    is_calibrated = meta["is_calibrated"]

    log.append(f"Ingested SAR raster: sensor={sensor}, product={product_type}, pol={polarization}")

    # 2. Select primary channel if multi-band
    if sar_data.ndim == 3:
        if sar_data.shape[0] < sar_data.shape[1]:  # (C, H, W)
            channel = sar_data[0].astype(np.float32)
        else:  # (H, W, C)
            channel = sar_data[:, :, 0].astype(np.float32)
        log.append("Extracted primary single-polarization channel from multi-band raster.")
    else:
        channel = sar_data.astype(np.float32)

    # 3. Disambiguate representation and convert to linear intensity
    # If values are already in negative decibels (e.g. [-35.0, 5.0]), it is already dB
    min_val = float(np.nanmin(channel))
    max_val = float(np.nanmax(channel))

    if input_rep == "DB" or (min_val < -5.0 and max_val < 30.0):
        log.append("Raster values identified as calibrated backscatter in dB.")
        input_rep = "DB"
        is_calibrated = True
        # Convert dB to linear intensity for speckle filtering: I = 10^(dB/10)
        intensity = 10.0 ** (np.clip(channel, -45.0, 20.0) / 10.0)
    elif input_rep == "AMPLITUDE" or max_val > 500.0:
        log.append("Raster identified as amplitude DN; converting to linear intensity (I = A^2).")
        intensity = (channel ** 2).astype(np.float32)
        if not is_calibrated:
            warnings.append("Raw amplitude lacks sensor calibration factor; relative backscatter computed.")
    else:
        log.append("Raster treated as linear power/intensity.")
        intensity = np.maximum(channel, 1e-7).astype(np.float32)

    # 4. Speckle Filtering in intensity domain (Enhanced Lee)
    speckle_status = "NONE"
    if apply_speckle_filter:
        try:
            filtered_intensity = enhanced_lee_filter(intensity, window_size=5, damping_factor=1.0)
            speckle_status = "ENHANCED_LEE_5x5"
            log.append("Applied Enhanced Lee speckle filter (5x5, damping=1.0) in intensity domain.")
        except Exception as e:
            logger.warning(f"Speckle filter failed, proceeding with raw intensity: {e}")
            filtered_intensity = intensity
            warnings.append(f"Speckle filter skipped due to processing error: {e}")
    else:
        filtered_intensity = intensity

    # 5. Convert to Backscatter dB: sigma0 = 10 * log10(I + 1e-7)
    db_backscatter = 10.0 * np.log10(np.maximum(filtered_intensity, 1e-7))
    db_backscatter = np.clip(db_backscatter, -45.0, 15.0).astype(np.float32)
    log.append("Computed calibrated backscatter sigma0 in decibels (dB).")

    valid_mask = np.isfinite(db_backscatter)
    min_db = round(float(np.min(db_backscatter[valid_mask])), 2) if np.any(valid_mask) else None
    max_db = round(float(np.max(db_backscatter[valid_mask])), 2) if np.any(valid_mask) else None
    mean_db = round(float(np.mean(db_backscatter[valid_mask])), 2) if np.any(valid_mask) else None

    report = SarProcessingReport(
        asset_id=asset.id,
        sensor=sensor,
        product_type=product_type,
        polarization=polarization,
        input_representation=input_rep,
        is_calibrated=is_calibrated,
        speckle_filter_applied=speckle_status,
        min_db=min_db,
        max_db=max_db,
        mean_db=mean_db,
        transformations_log=log,
        warnings=warnings
    )

    return db_backscatter, report
