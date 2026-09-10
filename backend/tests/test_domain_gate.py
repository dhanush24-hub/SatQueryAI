import pytest
from app.services.geospatial.domain_gate import assess_input_domain, DomainSuitability


def test_domain_gate_supported_optical():
    res = assess_input_domain(
        modality="OPTICAL",
        num_bands=3,
        dimensions=(256, 256),
        resolution_m=0.5,
        registration_status="HIGH",
        cloud_or_nodata_ratio=0.02
    )
    assert res.suitability == DomainSuitability.SUPPORTED
    assert res.is_suitable is True
    assert len(res.warnings) == 0


def test_domain_gate_rejects_sar():
    res = assess_input_domain(
        modality="SAR",
        num_bands=1,
        dimensions=(256, 256)
    )
    assert res.suitability == DomainSuitability.OUT_OF_DOMAIN
    assert res.is_suitable is False
    assert "SAR" in res.reasons[0]


def test_domain_gate_rejects_unacceptable_registration():
    res = assess_input_domain(
        modality="OPTICAL",
        num_bands=3,
        dimensions=(256, 256),
        registration_status="UNACCEPTABLE"
    )
    assert res.suitability == DomainSuitability.INSUFFICIENT_EVIDENCE
    assert res.is_suitable is False
    assert "registration" in res.reasons[0].lower()


def test_domain_gate_rejects_small_dimensions():
    res = assess_input_domain(
        modality="OPTICAL",
        num_bands=3,
        dimensions=(32, 32)
    )
    assert res.suitability == DomainSuitability.INSUFFICIENT_EVIDENCE
    assert res.is_suitable is False


def test_domain_gate_supported_with_warnings():
    res = assess_input_domain(
        modality="OPTICAL",
        num_bands=1,  # warning for single band
        dimensions=(256, 256),
        resolution_m=50.0  # warning for coarse resolution
    )
    assert res.suitability == DomainSuitability.SUPPORTED_WITH_WARNINGS
    assert res.is_suitable is True
    assert len(res.warnings) >= 2
