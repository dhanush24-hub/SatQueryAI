from app.domain.modalities import Modality, ImageFormat
from app.domain.tasks import TaskFamily, AnalysisStatus, EvidenceVerdict


def test_modality_enum_values():
    assert Modality.OPTICAL == "OPTICAL"
    assert Modality.MULTISPECTRAL == "MULTISPECTRAL"
    assert Modality.SAR == "SAR"
    assert Modality.UNKNOWN == "UNKNOWN"
    assert len(Modality) == 4


def test_image_format_enum_values():
    assert ImageFormat.GEOTIFF == "GEOTIFF"
    assert ImageFormat.TIFF == "TIFF"
    assert ImageFormat.PNG == "PNG"
    assert ImageFormat.JPEG == "JPEG"
    assert ImageFormat.WEBP == "WEBP"
    assert ImageFormat.UNKNOWN == "UNKNOWN"


def test_task_family_enum_values():
    expected_tasks = {
        "SINGLE_VQA",
        "SINGLE_GROUNDING",
        "SINGLE_CAPTION",
        "TEMPORAL_CHANGE",
        "TEMPORAL_CHANGE_VQA",
        "OPTICAL_SAR_ANALYSIS",
        "UNKNOWN",
    }
    assert {t.value for t in TaskFamily} == expected_tasks


def test_analysis_status_enum_values():
    expected_statuses = {
        "PENDING",
        "RUNNING",
        "COMPLETED",
        "COMPLETED_WITH_WARNINGS",
        "INSUFFICIENT_EVIDENCE",
        "VALIDATION_FAILED",
        "MODEL_UNAVAILABLE",
        "FAILED",
        "NOT_IMPLEMENTED",
    }
    assert {s.value for s in AnalysisStatus} == expected_statuses


def test_evidence_verdict_enum_values():
    expected_verdicts = {
        "supported",
        "partially_supported",
        "insufficient",
        "conflicting",
    }
    assert {v.value for v in EvidenceVerdict} == expected_verdicts
