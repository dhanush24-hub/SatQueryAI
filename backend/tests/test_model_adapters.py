import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.domain.modalities import Modality, ImageFormat
from app.schemas.image_asset import ImageAsset
from app.schemas.evidence import BoundingBox
from app.db.asset_repository import asset_repository
from app.registry.model_registry import model_tool_registry
from app.services.models.base import ModelAdapter
from app.services.models.schemas import VqaRequest, VqaResponse, GroundingRequest, GroundingResponse, GroundingBox
from app.services.models.coordinates import (
    model_box_to_pixel_box,
    pixel_box_to_canvas_box,
    pixel_box_to_geographic_bbox
)

client = TestClient(app)


def test_registry_resolution():
    """Verify registry resolves adapters for single-image tasks and returns None for future tasks."""
    vqa_adapter = model_tool_registry.get_adapter(TaskFamily.SINGLE_VQA)
    assert vqa_adapter is not None
    assert vqa_adapter.model_name == "Salesforce/blip-vqa-base"
    assert vqa_adapter.task_family == TaskFamily.SINGLE_VQA

    grounding_adapter = model_tool_registry.get_adapter(TaskFamily.SINGLE_GROUNDING)
    assert grounding_adapter is not None
    assert grounding_adapter.model_name == "google/owlvit-base-patch32"
    assert grounding_adapter.task_family == TaskFamily.SINGLE_GROUNDING

    # Temporal change is active in Prompt 5
    temp_adapter = model_tool_registry.get_adapter(TaskFamily.TEMPORAL_CHANGE)
    assert temp_adapter is not None
    assert temp_adapter.task_family == TaskFamily.TEMPORAL_CHANGE

    # Optical+SAR fusion is active in Prompt 8
    fusion_adapter = model_tool_registry.get_adapter(TaskFamily.OPTICAL_SAR_ANALYSIS)
    assert fusion_adapter is not None
    assert fusion_adapter.task_family == TaskFamily.OPTICAL_SAR_ANALYSIS
    # Unimplemented task (SINGLE_CAPTION) remains None
    assert model_tool_registry.get_adapter(TaskFamily.SINGLE_CAPTION) is None


def test_coordinate_conversion_and_clamping():
    """Verify coordinate mapping and boundary clamping from model space [0,1] to pixel and canvas percentages."""
    orig_w, orig_h = 1000, 500

    # Normal valid box: [0.1, 0.2, 0.5, 0.6] (xmin, ymin, xmax, ymax)
    raw_box = [0.1, 0.2, 0.5, 0.6]
    px_box = model_box_to_pixel_box(raw_box, orig_w, orig_h)
    assert px_box == [100, 100, 500, 300]

    canvas_box = pixel_box_to_canvas_box(px_box, orig_w, orig_h)
    assert canvas_box == [10.0, 20.0, 40.0, 40.0]

    # Out-of-bounds box: [-0.2, -0.1, 1.3, 1.5]
    oob_box = [-0.2, -0.1, 1.3, 1.5]
    px_oob = model_box_to_pixel_box(oob_box, orig_w, orig_h)
    assert px_oob == [0, 0, 1000, 500]

    canvas_oob = pixel_box_to_canvas_box(px_oob, orig_w, orig_h)
    assert canvas_oob == [0.0, 0.0, 100.0, 100.0]


def test_pixel_to_geo_conversion():
    """Verify affine geotransform projection for bounding boxes."""
    # Affine: a=10.0, b=0.0, c=100000.0, d=0.0, e=-10.0, f=200000.0
    affine_matrix = [10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0]
    px_box = [10, 20, 30, 50]

    geo = pixel_box_to_geographic_bbox(px_box, affine_matrix, "EPSG:4326")
    assert geo is not None
    # minx = 100000 + 10*10 = 100100
    # maxx = 100000 + 30*10 = 100300
    # maxy = 200000 + 20*(-10) = 199800
    # miny = 200000 + 50*(-10) = 199500
    assert geo[0] == 100100.0
    assert geo[1] == 199500.0
    assert geo[2] == 100300.0
    assert geo[3] == 199800.0


def test_bounding_box_pydantic_validation():
    """Verify BoundingBox schema strictly rejects negative or >100 percentage values."""
    valid_box = BoundingBox(x=10.5, y=20.0, width=30.0, height=40.0)
    assert valid_box.x == 10.5

    with pytest.raises(Exception):
        BoundingBox(x=-5.0, y=20.0, width=30.0, height=40.0)

    with pytest.raises(Exception):
        BoundingBox(x=10.0, y=105.0, width=30.0, height=40.0)


def test_analysis_with_invalid_asset_id():
    """API must return 404 when requested image asset ID is not in registry."""
    response = client.post(
        "/api/analysis",
        json={"query": "What land cover is visible?", "image_ids": ["non_existent_asset_123"]}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_sar_modality_rejection():
    """Optical specialists must explicitly reject SAR imagery with MODEL_UNAVAILABLE per modality rules."""
    sar_asset = ImageAsset(
        id="sar_test_asset_01",
        filename="sar_test_asset_01.tif",
        original_filename="sentinel1_grd.tif",
        storage_path="mock/path.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.SAR,
        width=256,
        height=256,
        bands=1,
        validation_status="valid"
    )
    asset_repository.save_asset(sar_asset, "mock/path.tif")

    response = client.post(
        "/api/analysis",
        json={"query": "What is the terrain type?", "image_ids": [sar_asset.id]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == AnalysisStatus.MODEL_UNAVAILABLE
    assert "sar" in data["answer"].lower()
    assert any("sar" in w.lower() for w in data["warnings"])
    assert data["confidence"] is None


class MockVqaAdapter(ModelAdapter):
    @property
    def model_name(self) -> str:
        return "mock_vqa_adapter"

    @property
    def task_family(self) -> TaskFamily:
        return TaskFamily.SINGLE_VQA

    @property
    def supported_modalities(self):
        return [Modality.OPTICAL, Modality.MULTISPECTRAL]

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self):
        pass

    def unload(self):
        pass

    def health(self):
        return {"status": "ready", "device": "cpu"}

    def metadata(self):
        return {"model_name": self.model_name}

    def predict(self, request: VqaRequest) -> VqaResponse:
        return VqaResponse(
            answer="Dense agricultural fields and a circular reservoir.",
            model_name="Salesforce/blip-vqa-base-mock",
            task_family=TaskFamily.SINGLE_VQA,
            device="cpu",
            inference_latency_ms=45.0,
            confidence=None,
            warnings=["Confidence null: Autoregressive VLM generation logits are uncalibrated per zero-fabrication rules."]
        )


class MockGroundingAdapter(ModelAdapter):
    @property
    def model_name(self) -> str:
        return "mock_grounding_adapter"

    @property
    def task_family(self) -> TaskFamily:
        return TaskFamily.SINGLE_GROUNDING

    @property
    def supported_modalities(self):
        return [Modality.OPTICAL, Modality.MULTISPECTRAL]

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self):
        pass

    def unload(self):
        pass

    def health(self):
        return {"status": "ready", "device": "cpu"}

    def metadata(self):
        return {"model_name": self.model_name}

    def predict(self, request: GroundingRequest) -> GroundingResponse:
        return GroundingResponse(
            boxes=[
                GroundingBox(
                    label="water body",
                    score=0.742,
                    canvas_box=[10.0, 15.0, 30.0, 25.0],
                    pixel_box=[50, 60, 200, 180],
                    geographic_bbox=None
                )
            ],
            model_name="google/owlvit-base-patch32-mock",
            task_family=TaskFamily.SINGLE_GROUNDING,
            device="cpu",
            inference_latency_ms=88.0,
            queries=request.queries,
            warnings=["Confidence null: Detector scores reflect raw sigmoid logits."]
        )


def test_vqa_execution_flow():
    """Verify single-image VQA executes and returns valid contract with uncalibrated confidence nullability."""
    test_asset = ImageAsset(
        id="opt_vqa_test_asset",
        filename="opt_vqa_test_asset.tif",
        original_filename="cartosat_scene.tif",
        storage_path="mock/path.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=500,
        height=400,
        bands=3,
        validation_status="valid"
    )
    asset_repository.save_asset(test_asset, "mock/path.tif")

    # Register mock adapter for test determinism
    mock_adapter = MockVqaAdapter()
    model_tool_registry.register_adapter(mock_adapter)

    response = client.post(
        "/api/analysis",
        json={"query": "What features are visible in this scene?", "image_ids": [test_asset.id]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == AnalysisStatus.COMPLETED
    assert data["task"] == TaskFamily.SINGLE_VQA
    assert "agricultural" in data["answer"]
    # Non-negotiable rule: uncalibrated generation logits must return confidence null
    assert data["confidence"] is None
    assert len(data["warnings"]) >= 1
    assert len(data["execution_summary"]) == 1
    step = data["execution_summary"][0]
    assert step["tool_name"] == "mock_vqa_adapter"
    assert step["status"] == "completed"


def test_grounding_execution_flow():
    """Verify text-guided grounding query is routed to SINGLE_GROUNDING and returns spatial evidence."""
    test_asset = ImageAsset(
        id="opt_grounding_test_asset",
        filename="opt_grounding_test_asset.tif",
        original_filename="sentinel2_rgb.tif",
        storage_path="mock/path.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=500,
        height=400,
        bands=3,
        validation_status="valid"
    )
    asset_repository.save_asset(test_asset, "mock/path.tif")

    # Register mock adapter for test determinism
    mock_adapter = MockGroundingAdapter()
    model_tool_registry.register_adapter(mock_adapter)

    response = client.post(
        "/api/analysis",
        json={"query": "Highlight the water body in this image.", "image_ids": [test_asset.id]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == AnalysisStatus.COMPLETED
    assert data["task"] == TaskFamily.SINGLE_GROUNDING
    assert len(data["evidence"]) == 1
    ev = data["evidence"][0]
    assert ev["evidence_type"] == "visual_region"
    assert ev["region"]["x"] == 10.0
    assert ev["region"]["y"] == 15.0
    assert ev["region"]["width"] == 30.0
    assert ev["region"]["height"] == 25.0
    assert ev["source_step"] == "SINGLE_GROUNDING"
    assert data["confidence"] is None
