from typing import Optional, Tuple
from shapely.geometry import box
from app.schemas.image_asset import ImageAsset
from app.schemas.compatibility import PairType, PairCompatibilityReport
from app.domain.modalities import Modality


def check_pair_compatibility(
    asset_a: ImageAsset,
    asset_b: ImageAsset,
    pair_type: PairType
) -> PairCompatibilityReport:
    """
    Evaluate geometric and radiometric compatibility between two ImageAssets.
    Does not assume co-registration solely from matching CRS or dimensions.
    """
    warnings = []
    recommendations = []

    # 1. CRS Analysis
    crs_a = asset_a.crs
    crs_b = asset_b.crs
    crs_match = bool(crs_a and crs_b and crs_a == crs_b)

    requires_reprojection = False
    if not crs_a or not crs_b:
        warnings.append("One or both images lack spatial reference systems (CRS); spatial overlap cannot be guaranteed.")
        requires_reprojection = True
    elif not crs_match:
        requires_reprojection = True
        warnings.append(f"Images have different coordinate reference systems ({crs_a} vs {crs_b}); reprojection is required.")
        recommendations.append(f"Run alignment service to reproject {asset_b.id} into {crs_a}.")

    # 2. Spatial Overlap Analysis via WGS84 Geographic Bounding Boxes
    spatial_overlap_percentage = 0.0
    has_spatial_overlap = False
    overlap_geographic_bbox = None

    if asset_a.geographic_bbox and asset_b.geographic_bbox:
        try:
            poly_a = box(*asset_a.geographic_bbox)
            poly_b = box(*asset_b.geographic_bbox)

            if poly_a.intersects(poly_b):
                inter = poly_a.intersection(poly_b)
                if not inter.is_empty and inter.area > 0:
                    min_area = min(poly_a.area, poly_b.area)
                    if min_area > 0:
                        spatial_overlap_percentage = round(float((inter.area / min_area) * 100.0), 2)
                        has_spatial_overlap = spatial_overlap_percentage > 0.01
                        bounds = inter.bounds
                        overlap_geographic_bbox = [round(b, 6) for b in bounds]
            else:
                has_spatial_overlap = False
                spatial_overlap_percentage = 0.0
                warnings.append("Images have completely disjoint spatial footprints; no geographic overlap detected.")
        except Exception as e:
            warnings.append(f"Error calculating spatial intersection: {str(e)}")
    else:
        warnings.append("Geographic bounding boxes missing; spatial overlap cannot be computed.")

    # 3. Resolution and Grid Analysis
    res_a = asset_a.resolution or 1.0
    res_b = asset_b.resolution or 1.0
    res_ratio = round(float(res_a / res_b if res_b != 0 else 1.0), 4)

    requires_resampling = abs(res_ratio - 1.0) > 0.05
    if requires_resampling:
        warnings.append(f"Pixel resolutions differ ({res_a}m vs {res_b}m, ratio {res_ratio}); grid resampling required.")
        recommendations.append("Resample to the finer resolution using bilinear or cubic interpolation.")

    # 4. Rigorous Co-registration Check
    # Important: Same CRS does NOT automatically mean co-registered.
    # Same dimensions do NOT automatically mean same geographic area.
    is_coregistered = False
    if (
        crs_match
        and asset_a.width == asset_b.width
        and asset_a.height == asset_b.height
        and asset_a.affine_transform
        and asset_b.affine_transform
    ):
        diffs = [abs(a - b) for a, b in zip(asset_a.affine_transform, asset_b.affine_transform)]
        if all(d < 1e-4 for d in diffs):
            is_coregistered = True
        else:
            warnings.append("Images share the same CRS and dimensions, but pixel grids are spatially shifted; fine co-registration required.")
    else:
        if crs_match and has_spatial_overlap:
            warnings.append("Images overlap in same CRS, but pixel grid bounds and dimensions differ.")

    # 5. Modality Compatibility
    modality_compatible = True
    if pair_type == PairType.OPTICAL_SAR:
        has_sar = asset_a.modality == Modality.SAR or asset_b.modality == Modality.SAR
        has_opt = asset_a.modality in [Modality.OPTICAL, Modality.MULTISPECTRAL] or asset_b.modality in [Modality.OPTICAL, Modality.MULTISPECTRAL]
        if not (has_sar and has_opt):
            modality_compatible = False
            warnings.append(
                f"Requested pair type is OPTICAL_SAR, but input modalities are {asset_a.modality.value} and {asset_b.modality.value}."
            )
            recommendations.append("Provide one Optical/Multispectral image and one SAR image for cross-sensor analysis.")

    elif pair_type == PairType.TEMPORAL:
        if (asset_a.modality == Modality.SAR and asset_b.modality != Modality.SAR) or (
            asset_a.modality != Modality.SAR and asset_b.modality == Modality.SAR
        ):
            warnings.append("Temporal comparison between Optical and SAR imagery introduces strong cross-sensor domain gap.")
            recommendations.append("Consider using OPTICAL_SAR pair analysis rather than bi-temporal change detection.")

    # Timestamps
    acq_dates = [asset_a.acquisition_time, asset_b.acquisition_time]
    if pair_type == PairType.TEMPORAL and not any(acq_dates):
        warnings.append("Acquisition timestamps are absent in raster metadata; temporal sequence order cannot be verified.")

    # Overall compatibility: requires at least some spatial overlap (or benchmark context)
    is_benchmark = (
        asset_a.validation_status == "benchmark" or
        asset_b.validation_status == "benchmark" or
        (isinstance(asset_a.metadata, dict) and asset_a.metadata.get("benchmark_mode")) or
        (isinstance(asset_b.metadata, dict) and asset_b.metadata.get("benchmark_mode"))
    )
    if is_benchmark:
        compatible = modality_compatible
    else:
        compatible = has_spatial_overlap and modality_compatible

    return PairCompatibilityReport(
        compatible=compatible,
        pair_type=pair_type,
        asset_id_a=asset_a.id,
        asset_id_b=asset_b.id,
        crs_a=crs_a,
        crs_b=crs_b,
        crs_match=crs_match,
        spatial_overlap_percentage=spatial_overlap_percentage,
        has_spatial_overlap=has_spatial_overlap,
        overlap_geographic_bbox=overlap_geographic_bbox,
        resolution_ratio=res_ratio,
        requires_reprojection=requires_reprojection,
        requires_resampling=requires_resampling,
        is_coregistered=is_coregistered,
        modality_compatible=modality_compatible,
        acquisition_dates=acq_dates,
        warnings=warnings,
        recommendations=recommendations,
    )
