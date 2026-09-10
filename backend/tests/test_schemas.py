import pytest
from pydantic import ValidationError
from app.domain.modalities import Modality, ImageFormat
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.schemas.image_asset import ImageAsset
from app.schemas.execution import ExecutionStep
from app.schemas.evidence import BoundingBox, EvidenceItem
from app.schemas.analysis import AnalysisRequest, AnalysisResult


def test_image_asset_valid():
    asset = ImageAsset(
        id="img_123",
        filename="test.tif",
        storage_path="/path/to/test.tif",
        original_filename="input.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.MULTISPECTRAL,
        width=1024,
        height=1024,
        bands=4,
        crs="EPSG:32644",
        resolution=10.0,
        bbox=[78.0, 12.0, 78.5, 12.5],
        nodata=0.0,
        acquisition_time="2026-03-01T10:00:00Z",
        sensor="Sentinel-2",
        metadata={"platform": "S2A"},
        validation_status="valid",
        warnings=[]
    )
    assert asset.id == "img_123"
    assert asset.format == ImageFormat.GEOTIFF
    assert asset.modality == Modality.MULTISPECTRAL
    assert asset.width == 1024


def test_bounding_box_validation():
    # Valid bounding box within 0-100 percentage
    box = BoundingBox(x=10.5, y=20.0, width=30.0, height=40.0)
    assert box.x == 10.5
    assert box.width == 30.0

    # Invalid: coordinates exceeding 100%
    with pytest.raises(ValidationError):
        BoundingBox(x=-1.0, y=20.0, width=30.0, height=40.0)

    with pytest.raises(ValidationError):
        BoundingBox(x=10.0, y=20.0, width=150.0, height=40.0)


def test_analysis_request_validation():
    # Valid request
    req = AnalysisRequest(
        query="What change occurred in this forest area?",
        image_ids=["img_1", "img_2"]
    )
    assert req.query == "What change occurred in this forest area?"
    assert len(req.image_ids) == 2

    # Query too short
    with pytest.raises(ValidationError):
        AnalysisRequest(query="", image_ids=["img_1"])

    # Too many images (max 6)
    with pytest.raises(ValidationError):
        AnalysisRequest(query="Question", image_ids=[f"img_{i}" for i in range(7)])


def test_observable_execution_step():
    step = ExecutionStep(
        step="Optical Land Cover Assessment",
        tool_name="optical_vqa_adapter",
        model_name="remoteclip-vit-b",
        parameters={"threshold": 0.85},
        status="completed",
        duration_ms=350,
        summary="Identified mangrove wetland and tidal channel features."
    )
    assert step.step == "Optical Land Cover Assessment"
    assert step.status == "completed"
    assert step.duration_ms == 350
    # Internal hidden CoT cannot be passed because model has no reasoning fields
    assert not hasattr(step, "chain_of_thought")
    assert not hasattr(step, "internal_reasoning")


def test_analysis_result_schema():
    result = AnalysisResult(
        id="anl_001",
        answer="Mangrove canopy density remained stable across the observation.",
        task=TaskFamily.SINGLE_VQA,
        confidence="Qualitative visual support · not calibrated",
        evidence=[],
        warnings=["No ground truth calibration available."],
        execution_summary=[],
        status=AnalysisStatus.COMPLETED
    )
    assert result.id == "anl_001"
    assert result.status == AnalysisStatus.COMPLETED
    assert result.task == TaskFamily.SINGLE_VQA
