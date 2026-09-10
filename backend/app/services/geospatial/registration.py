import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from app.core.logging import logger


class RegistrationAssessment(BaseModel):
    """Observable registration verification report."""
    method: str = "2D_phase_correlation_and_zncc"
    shift_x_pixels: float = 0.0
    shift_y_pixels: float = 0.0
    shift_magnitude_pixels: float = 0.0
    correlation_coefficient: float = 1.0
    quality_status: str = "HIGH"  # HIGH, ACCEPTABLE, POOR, UNVERIFIED
    valid_overlap_pct: float = 100.0
    is_sufficient_for_pixel_localization: bool = True
    warnings: List[str] = Field(default_factory=list)


def verify_registration_quality(
    img1: np.ndarray,
    img2: np.ndarray,
    nodata_val: Optional[float] = None
) -> RegistrationAssessment:
    """
    Evaluates empirical spatial co-registration between two aligned rasters using
    2D Fourier Phase Correlation and Zero-mean Normalized Cross Correlation (ZNCC).
    
    Validates physical ground feature alignment beyond simple coordinate grid metadata.
    """
    warnings: List[str] = []

    # 1. Convert to 2D single-channel grayscale if multi-band
    def to_grayscale(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            # If shape is (bands, H, W)
            if arr.shape[0] in [1, 3, 4] and arr.shape[0] < arr.shape[1]:
                if arr.shape[0] >= 3:
                    # ITU-R BT.601 luma weights
                    return 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
                return arr[0].astype(np.float32)
            # If shape is (H, W, C)
            elif arr.shape[2] in [1, 3, 4]:
                if arr.shape[2] >= 3:
                    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                return arr[:, :, 0].astype(np.float32)
        return arr.astype(np.float32)

    g1 = to_grayscale(img1)
    g2 = to_grayscale(img2)

    # 2. Ensure matching dimensions
    min_h = min(g1.shape[0], g2.shape[0])
    min_w = min(g1.shape[1], g2.shape[1])
    if min_h < 16 or min_w < 16:
        return RegistrationAssessment(
            quality_status="UNVERIFIED",
            correlation_coefficient=0.0,
            is_sufficient_for_pixel_localization=False,
            warnings=["Scene dimensions too small (<16x16) for empirical registration assessment."]
        )

    g1 = g1[:min_h, :min_w]
    g2 = g2[:min_h, :min_w]

    # 3. Handle nodata / valid masks
    valid_mask = np.isfinite(g1) & np.isfinite(g2)
    if nodata_val is not None:
        valid_mask &= (g1 != nodata_val) & (g2 != nodata_val)

    total_pixels = min_h * min_w
    valid_pixels = int(np.sum(valid_mask))
    valid_pct = round((valid_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

    if valid_pct < 20.0:
        return RegistrationAssessment(
            valid_overlap_pct=valid_pct,
            quality_status="POOR",
            correlation_coefficient=0.0,
            is_sufficient_for_pixel_localization=False,
            warnings=[f"Insufficient valid data overlap ({valid_pct}% < 20%). Registration cannot be verified."]
        )

    # Fast-path for identical/nearly-identical scenes (e.g. baseline zero-change test)
    if np.allclose(g1[valid_mask], g2[valid_mask], atol=1e-3):
        return RegistrationAssessment(
            shift_x_pixels=0.0,
            shift_y_pixels=0.0,
            shift_magnitude_pixels=0.0,
            correlation_coefficient=1.0,
            quality_status="HIGH",
            valid_overlap_pct=valid_pct,
            is_sufficient_for_pixel_localization=True,
            warnings=[]
        )

    # Fill invalid pixels with scene mean for stable FFT
    mean1 = float(np.mean(g1[valid_mask])) if valid_pixels > 0 else 0.0
    mean2 = float(np.mean(g2[valid_mask])) if valid_pixels > 0 else 0.0
    g1_filled = np.where(valid_mask, g1, mean1)
    g2_filled = np.where(valid_mask, g2, mean2)

    # 4. Phase Correlation via 2D FFT
    # Window function to reduce spectral leakage
    hanning_2d = np.outer(np.hanning(min_h), np.hanning(min_w))
    f1 = np.fft.fft2(g1_filled * hanning_2d)
    f2 = np.fft.fft2(g2_filled * hanning_2d)

    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    correlation_surface = np.real(np.fft.ifft2(cross_power))

    # Shift zero-frequency component to center
    correlation_surface = np.fft.fftshift(correlation_surface)

    # Peak detection
    center_y, center_x = min_h // 2, min_w // 2
    max_idx = np.unravel_index(np.argmax(correlation_surface), correlation_surface.shape)
    shift_y = float(max_idx[0] - center_y)
    shift_x = float(max_idx[1] - center_x)
    shift_mag = float(np.sqrt(shift_x**2 + shift_y**2))

    # Peak correlation value (measure of structural similarity)
    peak_val = float(correlation_surface[max_idx])

    # 5. Zero-mean Normalized Cross-Correlation (ZNCC) on valid overlapping pixels
    v1 = g1[valid_mask] - mean1
    v2 = g2[valid_mask] - mean2
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-10
    zncc = float(np.dot(v1, v2) / denom)
    zncc = max(-1.0, min(1.0, zncc))

    # 6. Quality Status Determination
    # Reprojection aligns coordinate systems, but uncalibrated parallax or GCP error causes residual shift
    if shift_mag <= 1.5 and zncc >= 0.40:
        quality_status = "HIGH"
    elif shift_mag <= 3.5 and zncc >= 0.20:
        quality_status = "ACCEPTABLE"
        warnings.append(f"Minor co-registration residual detected: {shift_mag:.2f} px shift (ZNCC: {zncc:.2f}).")
    else:
        quality_status = "POOR"
        warnings.append(
            f"Significant co-registration residual: {shift_mag:.2f} px shift (ZNCC: {zncc:.2f}). "
            "Pixel-level change localization may contain false alarms along high-contrast edges."
        )

    is_sufficient = quality_status in ["HIGH", "ACCEPTABLE"]

    logger.info(
        f"Registration Assessment: status={quality_status}, shift=({shift_x:.1f}, {shift_y:.1f})px, "
        f"ZNCC={zncc:.3f}, overlap={valid_pct}%"
    )

    return RegistrationAssessment(
        shift_x_pixels=round(shift_x, 2),
        shift_y_pixels=round(shift_y, 2),
        shift_magnitude_pixels=round(shift_mag, 2),
        correlation_coefficient=round(zncc, 4),
        quality_status=quality_status,
        valid_overlap_pct=valid_pct,
        is_sufficient_for_pixel_localization=is_sufficient,
        warnings=warnings
    )
