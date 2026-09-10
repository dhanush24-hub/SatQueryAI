import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.domain.modalities import Modality, ImageFormat
from app.schemas.image_asset import ImageAsset
from app.schemas.analysis import AnalysisRequest
from app.db.asset_repository import asset_repository
from app.registry.model_registry import model_tool_registry
from app.services.orchestration import (
    TaskClassifier,
    WorkflowPlanner,
    PlanValidator,
    WorkflowExecutor,
    EvidenceAssessor,
    WorkflowController,
    workflow_controller,
    WorkflowPlan,
    PlanStep
)
from app.services.models.base import ModelAdapter
from app.services.models.schemas import VqaRequest, VqaResponse, GroundingRequest, GroundingResponse, GroundingBox

client = TestClient(app)

task_classifier = TaskClassifier()
workflow_planner = WorkflowPlanner()
plan_validator = PlanValidator()
evidence_assessor = EvidenceAssessor()


# Mock Adapters for isolated, deterministic controller testing
class ControllerMockVqaAdapter(ModelAdapter):
    @property
    def model_name(self) -> str:
        return "Salesforce/blip-vqa-base"

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
            answer="Agricultural crop patterns with an active canal.",
            model_name=self.model_name,
            task_family=TaskFamily.SINGLE_VQA,
            device="cpu",
            inference_latency_ms=30.0,
            confidence=None,
            warnings=["Uncalibrated logits."]
        )


class ControllerMockGroundingAdapter(ModelAdapter):
    def __init__(self, boxes=None):
        self._boxes = boxes if boxes is not None else [
            GroundingBox(
                label="water canal",
                score=0.65,
                canvas_box=[10.0, 10.0, 30.0, 30.0],
                pixel_box=[50, 50, 150, 150],
                geographic_bbox=None
            )
        ]

    @property
    def model_name(self) -> str:
        return "google/owlvit-base-patch32"

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
            boxes=self._boxes,
            model_name=self.model_name,
            task_family=TaskFamily.SINGLE_GROUNDING,
            device="cpu",
            inference_latency_ms=40.0,
            queries=request.queries,
            warnings=["Raw sigmoid logits."]
        )


class FailingMockAdapter(ModelAdapter):
    @property
    def model_name(self) -> str:
        return "Salesforce/blip-vqa-base"

    @property
    def task_family(self) -> TaskFamily:
        return TaskFamily.SINGLE_VQA

    @property
    def supported_modalities(self):
        return [Modality.OPTICAL]

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self):
        pass

    def unload(self):
        pass

    def health(self):
        return {"status": "error", "device": "cpu"}

    def metadata(self):
        return {"model_name": self.model_name}

    def predict(self, request: VqaRequest) -> VqaResponse:
        raise RuntimeError("Specialist model inference pipeline crashed on host.")


# =========================================================================
# 1. Query Interpretation and Task Classification Tests
# =========================================================================

def test_query_classification_single_vqa():
    """Verify natural language queries requesting descriptions classify as SINGLE_VQA."""
    single_opt = [ImageAsset(id="a1", filename="a.tif", original_filename="a.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)]
    
    queries = [
        "What is visible in this satellite scene?",
        "Describe the dominant land cover.",
        "Is there evidence of flood inundation?",
        "Identify the structures present in the image."
    ]
    for q in queries:
        task, intent = task_classifier.classify(q, single_opt)
        assert task == TaskFamily.SINGLE_VQA
        assert intent["needs_vqa"] is True


def test_query_classification_single_grounding():
    """Verify natural language queries requesting spatial localization classify as SINGLE_GROUNDING."""
    single_opt = [ImageAsset(id="a1", filename="a.tif", original_filename="a.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)]
    
    queries = [
        "Locate the water body in the image.",
        "Highlight the agricultural fields.",
        "Where is the building complex?",
        "Find and box all solar panels.",
        "Detect roads across the scene."
    ]
    for q in queries:
        task, intent = task_classifier.classify(q, single_opt)
        assert task == TaskFamily.SINGLE_GROUNDING
        assert intent["needs_grounding"] is True


def test_query_classification_composite_vqa_and_grounding():
    """Verify queries asking to both describe and locate trigger composite plan."""
    single_opt = [ImageAsset(id="a1", filename="a.tif", original_filename="a.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)]
    
    query = "What is the vegetation type and locate the water body?"
    task, intent = task_classifier.classify(query, single_opt)
    assert intent.get("is_composite") is True
    assert intent["needs_vqa"] is True
    assert intent["needs_grounding"] is True


def test_query_classification_temporal_tasks():
    """Verify multi-temporal asset configurations classify as TEMPORAL_CHANGE or TEMPORAL_CHANGE_VQA."""
    two_opts = [
        ImageAsset(id="t1", filename="t1.tif", original_filename="t1.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3),
        ImageAsset(id="t2", filename="t2.tif", original_filename="t2.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)
    ]
    
    task1, _ = task_classifier.classify("What changed between these two dates", two_opts)
    assert task1 == TaskFamily.TEMPORAL_CHANGE

    task2, _ = task_classifier.classify("Has the built-up area increased between t1 and t2?", two_opts)
    assert task2 == TaskFamily.TEMPORAL_CHANGE_VQA


def test_query_classification_optical_sar_fusion():
    """Verify Optical + SAR pair configuration classifies as OPTICAL_SAR_ANALYSIS."""
    opt_sar_pair = [
        ImageAsset(id="opt1", filename="opt.tif", original_filename="opt.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3),
        ImageAsset(id="sar1", filename="sar.tif", original_filename="sar.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.SAR, width=100, height=100, bands=1)
    ]
    
    task, _ = task_classifier.classify("Analyze the optical and SAR imagery for flood detection", opt_sar_pair)
    assert task == TaskFamily.OPTICAL_SAR_ANALYSIS


def test_unknown_modality_handling():
    """Verify assets with UNKNOWN modality are safely handled in classification."""
    unknown_asset = [ImageAsset(id="u1", filename="u.tif", original_filename="u.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.UNKNOWN, width=100, height=100, bands=1)]
    task, intent = task_classifier.classify("What is here?", unknown_asset)
    assert task == TaskFamily.SINGLE_VQA


# =========================================================================
# 2. Plan Construction and Validation Tests
# =========================================================================

def test_plan_construction_single_vqa():
    """Verify plan construction creates a single VQA step for single-image VQA queries."""
    single_opt = [ImageAsset(id="a1", filename="a.tif", original_filename="a.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)]
    task, intent = task_classifier.classify("What is visible?", single_opt)
    plan = workflow_planner.create_plan(
        task_family=task,
        query="What is visible?",
        assets=single_opt,
        classification_meta=intent,
        parameters={}
    )
    
    assert plan.task_family == TaskFamily.SINGLE_VQA
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_id == "adapter_single_vqa"
    assert "answer" in plan.expected_outputs


def test_plan_construction_composite_vqa_grounding():
    """Verify plan construction creates sequential steps with dependencies for composite queries."""
    single_opt = [ImageAsset(id="a1", filename="a.tif", original_filename="a.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)]
    task, intent = task_classifier.classify("What crops are visible and locate the water body?", single_opt)
    plan = workflow_planner.create_plan(
        task_family=task,
        query="What crops are visible and locate the water body?",
        assets=single_opt,
        classification_meta=intent,
        parameters={}
    )
    
    assert len(plan.steps) == 2
    step_tools = [s.tool_id for s in plan.steps]
    assert "adapter_single_vqa" in step_tools
    assert "adapter_single_grounding" in step_tools
    assert plan.steps[1].dependencies == [plan.steps[0].step_id]


def test_plan_validation_cycle_detection():
    """Verify planner validator rejects cyclic dependencies."""
    step1 = PlanStep(
        step_id="step_1",
        tool_id="adapter_single_vqa",
        task_family=TaskFamily.SINGLE_VQA,
        description="Step 1",
        input_bindings={"asset_id": "a1"},
        parameters={},
        dependencies=["step_2"]
    )
    step2 = PlanStep(
        step_id="step_2",
        tool_id="adapter_single_grounding",
        task_family=TaskFamily.SINGLE_GROUNDING,
        description="Step 2",
        input_bindings={"asset_id": "a1"},
        parameters={},
        dependencies=["step_1"]
    )
    
    cyclic_plan = WorkflowPlan(
        plan_id="plan_cyclic",
        task_family=TaskFamily.SINGLE_VQA,
        input_asset_ids=["a1"],
        steps=[step1, step2],
        dependencies={
            "step_1": ["step_2"],
            "step_2": ["step_1"]
        },
        permitted_parameters={},
        expected_outputs=["answer"]
    )
    is_valid, error_msg, status = plan_validator.validate_plan(
        cyclic_plan,
        [ImageAsset(id="a1", filename="a.tif", original_filename="a.tif", storage_path="", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)]
    )
    assert is_valid is False
    assert "cyclic" in error_msg.lower()


def test_plan_validation_unavailable_tool_rejection():
    """Verify plans containing unavailable tools (e.g., adapter_single_caption) are rejected before execution."""
    step_caption = PlanStep(
        step_id="step_caption",
        tool_id="adapter_single_caption",
        task_family=TaskFamily.SINGLE_CAPTION,
        description="Single image captioning step",
        input_bindings={"asset_id": "opt1"},
        parameters={},
        dependencies=[]
    )
    plan = WorkflowPlan(
        plan_id="plan_caption",
        task_family=TaskFamily.SINGLE_CAPTION,
        input_asset_ids=["opt1"],
        steps=[step_caption],
        dependencies={"step_caption": []},
        permitted_parameters={},
        expected_outputs=["caption"]
    )
    single_asset = [
        ImageAsset(id="opt1", filename="opt1.tif", original_filename="opt1.tif", storage_path="mock/opt1.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)
    ]
    is_valid, error_msg, status = plan_validator.validate_plan(plan, single_asset)
    assert is_valid is False
    assert "unsupported" in error_msg.lower() or "unavailable" in error_msg.lower()
    assert status == AnalysisStatus.MODEL_UNAVAILABLE


def test_plan_validation_parameter_bounds():
    """Verify parameters outside bounded ranges are rejected."""
    step_grounding = PlanStep(
        step_id="step_g",
        tool_id="adapter_single_grounding",
        task_family=TaskFamily.SINGLE_GROUNDING,
        description="Grounding step",
        input_bindings={"asset_id": "a1"},
        parameters={"threshold": 2.5},  # Max allowed is 1.0
        dependencies=[]
    )
    plan = WorkflowPlan(
        plan_id="plan_invalid_params",
        task_family=TaskFamily.SINGLE_GROUNDING,
        input_asset_ids=["a1"],
        steps=[step_grounding],
        dependencies={"step_g": []},
        permitted_parameters={},
        expected_outputs=["detections"]
    )
    single_opt = [ImageAsset(id="a1", filename="a.tif", original_filename="a.tif", storage_path="mock/a1.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3)]
    is_valid, error_msg, status = plan_validator.validate_plan(plan, single_opt)
    assert is_valid is False
    assert "above allowed maximum" in error_msg.lower()


# =========================================================================
# 3. Evidence Assessment Gate & Scientific Integrity Tests
# =========================================================================

def test_evidence_assessor_weak_evidence():
    """Verify detector scores < 0.05 are classified as WEAK evidence with low reliability."""
    weak_box = GroundingBox(
        label="submerged road",
        score=0.035,  # Low score typical of generic model on remote sensing
        canvas_box=[20.0, 20.0, 15.0, 15.0],
        pixel_box=[100, 100, 175, 175],
        geographic_bbox=None
    )
    grounding_res = GroundingResponse(
        boxes=[weak_box],
        model_name="mock_owlvit",
        task_family=TaskFamily.SINGLE_GROUNDING,
        device="cpu",
        inference_latency_ms=25.0,
        queries=["submerged road"],
        warnings=[]
    )
    items, assessment = evidence_assessor.assess_grounding_evidence(grounding_res, "asset_test", (500, 500))
    assert assessment.evidence_status == "WEAK"
    assert any("WEAK_DETECTION_SCORE" in flag for flag in assessment.quality_flags)
    assert len(items) == 1
    assert items[0].reliability == "low"


def test_evidence_assessor_empty_evidence():
    """Verify zero detections return INSUFFICIENT evidence."""
    grounding_res = GroundingResponse(
        boxes=[],
        model_name="mock_owlvit",
        task_family=TaskFamily.SINGLE_GROUNDING,
        device="cpu",
        inference_latency_ms=20.0,
        queries=["missing object"],
        warnings=[]
    )
    items, assessment = evidence_assessor.assess_grounding_evidence(grounding_res, "asset_test", (500, 500))
    assert assessment.evidence_status == "INSUFFICIENT"
    assert len(items) == 0
    assert assessment.confidence is None


def test_evidence_assessor_degenerate_box_filtering():
    """Verify scene-dominating boxes (>= 85% area) are filtered out as degenerate false alarms."""
    degenerate_box = GroundingBox(
        label="flooded area",
        score=0.45,
        canvas_box=[0.0, 0.0, 95.0, 95.0],  # ~90% of scene
        pixel_box=[0, 0, 475, 475],
        geographic_bbox=None
    )
    grounding_res = GroundingResponse(
        boxes=[degenerate_box],
        model_name="mock_owlvit",
        task_family=TaskFamily.SINGLE_GROUNDING,
        device="cpu",
        inference_latency_ms=30.0,
        queries=["flooded area"],
        warnings=[]
    )
    items, assessment = evidence_assessor.assess_grounding_evidence(grounding_res, "asset_test", (500, 500))
    assert len(items) == 0
    assert any("DEGENERATE_GLOBAL_BOX_EXCLUDED" in flag for flag in assessment.quality_flags)


def test_evidence_assessor_micro_box_filtering():
    """Verify micro point-noise boxes (< 0.03% area) are filtered out."""
    micro_box = GroundingBox(
        label="small pixel anomaly",
        score=0.50,
        canvas_box=[10.0, 10.0, 0.1, 0.1],  # ~0.01% area
        pixel_box=[50, 50, 51, 51],
        geographic_bbox=None
    )
    grounding_res = GroundingResponse(
        boxes=[micro_box],
        model_name="mock_owlvit",
        task_family=TaskFamily.SINGLE_GROUNDING,
        device="cpu",
        inference_latency_ms=30.0,
        queries=["small pixel anomaly"],
        warnings=[]
    )
    items, assessment = evidence_assessor.assess_grounding_evidence(grounding_res, "asset_test", (500, 500))
    assert len(items) == 0
    assert any("MICRO_NOISE_EXCLUDED" in flag for flag in assessment.quality_flags)


def test_evidence_assessor_nms_suppression():
    """Verify Non-Maximum Suppression suppresses highly overlapping redundant boxes (IoU > 0.60)."""
    box1 = GroundingBox(
        label="building",
        score=0.75,
        canvas_box=[10.0, 10.0, 20.0, 20.0],
        pixel_box=[50, 50, 150, 150],
        geographic_bbox=None
    )
    box2 = GroundingBox(
        label="building",
        score=0.60,
        canvas_box=[11.0, 11.0, 19.0, 19.0],  # ~85% overlap with box1
        pixel_box=[55, 55, 150, 150],
        geographic_bbox=None
    )
    grounding_res = GroundingResponse(
        boxes=[box1, box2],
        model_name="mock_owlvit",
        task_family=TaskFamily.SINGLE_GROUNDING,
        device="cpu",
        inference_latency_ms=30.0,
        queries=["building"],
        warnings=[]
    )
    items, assessment = evidence_assessor.assess_grounding_evidence(grounding_res, "asset_test", (500, 500))
    assert len(items) == 1
    assert "building" in items[0].description
    assert items[0].region.x == 10.0


# =========================================================================
# 4. Controller End-to-End Orchestration & Trace Observable Tests
# =========================================================================

def test_controller_vqa_and_observable_trace():
    """Verify controller executes single-image VQA and returns observable trace without private CoT."""
    asset = ImageAsset(
        id="ctrl_opt_vqa_001",
        filename="ctrl_vqa.tif",
        original_filename="ctrl_vqa.tif",
        storage_path="mock/path.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=500,
        height=500,
        bands=3,
        validation_status="valid"
    )
    asset_repository.save_asset(asset, "mock/path.tif")
    model_tool_registry.register_adapter(ControllerMockVqaAdapter())

    req = AnalysisRequest(query="What land cover is visible?", image_ids=[asset.id])
    result = workflow_controller.execute_analysis(req)
    assert result.status == AnalysisStatus.COMPLETED
    assert result.task == TaskFamily.SINGLE_VQA
    assert "Agricultural" in result.answer
    assert result.confidence is None
    assert len(result.execution_summary) == 1
    
    trace_step = result.execution_summary[0]
    assert trace_step.status == "completed"
    assert trace_step.tool_name in ["Salesforce/blip-vqa-base", "adapter_single_vqa"]
    assert trace_step.duration_ms >= 0


def test_controller_composite_execution_order():
    """Verify composite query runs VQA then Grounding sequentially with observable traces."""
    asset = ImageAsset(
        id="ctrl_opt_comp_001",
        filename="ctrl_comp.tif",
        original_filename="ctrl_comp.tif",
        storage_path="mock/path.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=500,
        height=500,
        bands=3,
        validation_status="valid"
    )
    asset_repository.save_asset(asset, "mock/path.tif")
    model_tool_registry.register_adapter(ControllerMockVqaAdapter())
    model_tool_registry.register_adapter(ControllerMockGroundingAdapter())

    req = AnalysisRequest(query="What crops are visible and locate the water body?", image_ids=[asset.id])
    result = workflow_controller.execute_analysis(req)
    assert result.status == AnalysisStatus.COMPLETED
    assert len(result.execution_summary) == 2
    assert result.execution_summary[0].tool_name in ["Salesforce/blip-vqa-base", "adapter_single_vqa"]
    assert result.execution_summary[1].tool_name in ["google/owlvit-base-patch32", "adapter_single_grounding"]
    assert len(result.evidence) == 1
    assert "water canal" in result.evidence[0].description


def test_controller_specialist_failure_propagation():
    """Verify specialist failure sets FAILED status and records error in observable trace."""
    asset = ImageAsset(
        id="ctrl_opt_fail_001",
        filename="ctrl_fail.tif",
        original_filename="ctrl_fail.tif",
        storage_path="mock/path.tif",
        format=ImageFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=500,
        height=500,
        bands=3,
        validation_status="valid"
    )
    asset_repository.save_asset(asset, "mock/path.tif")
    model_tool_registry.register_adapter(FailingMockAdapter())

    req = AnalysisRequest(query="What is visible?", image_ids=[asset.id])
    result = workflow_controller.execute_analysis(req)
    assert result.status == AnalysisStatus.FAILED
    assert "crashed" in result.answer.lower()
    assert any("crashed" in w.lower() for w in result.warnings)
    assert result.execution_summary[0].status == "failed"


def test_controller_temporal_task_rejects_unreferenced_images():
    """Verify requesting temporal change analysis on unreferenced images returns VALIDATION_FAILED."""
    t1 = ImageAsset(id="ctrl_t1", filename="t1.tif", original_filename="t1.tif", storage_path="mock/t1.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3, validation_status="valid")
    t2 = ImageAsset(id="ctrl_t2", filename="t2.tif", original_filename="t2.tif", storage_path="mock/t2.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3, validation_status="valid")
    asset_repository.save_asset(t1, "mock/t1.tif")
    asset_repository.save_asset(t2, "mock/t2.tif")

    response = client.post("/api/analysis", json={"query": "What changed between these images?", "image_ids": [t1.id, t2.id]})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == AnalysisStatus.VALIDATION_FAILED
    assert "incompatible" in data["answer"].lower() or "validation" in data["answer"].lower()
    assert len(data["evidence"]) == 0


def test_controller_unsupported_optical_sar_task():
    """Verify requesting Optical+SAR fusion without valid georeferencing fails server-side validation."""
    opt = ImageAsset(id="ctrl_opt", filename="opt.tif", original_filename="opt.tif", storage_path="mock/opt.tif", format=ImageFormat.GEOTIFF, modality=Modality.OPTICAL, width=100, height=100, bands=3, validation_status="valid")
    sar = ImageAsset(id="ctrl_sar", filename="sar.tif", original_filename="sar.tif", storage_path="mock/sar.tif", format=ImageFormat.GEOTIFF, modality=Modality.SAR, width=100, height=100, bands=1, validation_status="valid")
    asset_repository.save_asset(opt, "mock/opt.tif")
    asset_repository.save_asset(sar, "mock/sar.tif")

    response = client.post("/api/analysis", json={"query": "Extract water from optical and SAR", "image_ids": [opt.id, sar.id]})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == AnalysisStatus.VALIDATION_FAILED
    assert "incompatible" in data["answer"].lower() or "validation" in data["answer"].lower()
