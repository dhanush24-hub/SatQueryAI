from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.domain.tasks import TaskFamily
from app.domain.modalities import Modality
from app.core.logging import logger


class ToolDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique tool identifier")
    tool_id: Optional[str] = Field(None, description="Standard tool identifier (defaults to name)")
    task_types: List[TaskFamily] = Field(..., description="Task families this tool supports")
    modalities: List[Modality] = Field(..., description="Supported sensor modalities")
    model_name: Optional[str] = Field(None, description="Underlying model checkpoint / architecture")
    version: str = Field(default="0.1.0", description="Tool or model version")
    enabled: bool = Field(default=True, description="Whether the tool is active")
    availability: str = Field(default="AVAILABLE", description="Availability state: AVAILABLE | UNAVAILABLE")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Typed input parameter schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Typed output schema")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Compute and memory requirements")
    permitted_parameters: Dict[str, Any] = Field(default_factory=dict, description="Allowlisted parameter bounds")
    callable_ref: Optional[Any] = Field(default=None, exclude=True, description="Runtime executable function or class")

    @property
    def id(self) -> str:
        return self.tool_id or self.name

    @property
    def supported_tasks(self) -> List[TaskFamily]:
        return self.task_types

    @property
    def supported_modalities(self) -> List[Modality]:
        return self.modalities


class ModelToolRegistry:
    """
    Central registry for specialist remote sensing models and analysis tools.
    Provides lazy loading, task-based resolution, explicit adapter lifecycle,
    and capability metadata for the agentic controller.
    """
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._adapters: Dict[TaskFamily, Any] = {}

    def _register_default_tool_metadata(self) -> None:
        """Register capability metadata for all planned system tools."""
        # 1. Single-Image VQA (Available)
        self.register_tool(
            ToolDefinition(
                name="adapter_single_vqa",
                tool_id="adapter_single_vqa",
                task_types=[TaskFamily.SINGLE_VQA],
                modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
                model_name="Salesforce/blip-vqa-base",
                version="1.0.0",
                enabled=True,
                availability="AVAILABLE",
                input_schema={"asset_id": "str", "question": "str"},
                output_schema={"answer": "str", "device": "str", "latency_ms": "float"},
                resource_requirements={"min_ram_gb": 3.0, "accelerator": "mps/cuda/cpu"},
                permitted_parameters={"max_new_tokens": {"type": "int", "min": 1, "max": 64}}
            )
        )

        # 2. Single-Image Grounding (Available)
        self.register_tool(
            ToolDefinition(
                name="adapter_single_grounding",
                tool_id="adapter_single_grounding",
                task_types=[TaskFamily.SINGLE_GROUNDING],
                modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
                model_name="google/owlvit-base-patch32",
                version="1.0.0",
                enabled=True,
                availability="AVAILABLE",
                input_schema={"asset_id": "str", "queries": "List[str]", "threshold": "float"},
                output_schema={"boxes": "List[GroundingBox]", "device": "str", "latency_ms": "float"},
                resource_requirements={"min_ram_gb": 2.0, "accelerator": "mps/cuda/cpu"},
                permitted_parameters={"threshold": {"type": "float", "min": 0.001, "max": 1.0}}
            )
        )

        # 3. Single-Image Captioning (Planned - Prompt 5)
        self.register_tool(
            ToolDefinition(
                name="adapter_single_caption",
                tool_id="adapter_single_caption",
                task_types=[TaskFamily.SINGLE_CAPTION],
                modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
                model_name="unassigned/rs-caption-base",
                version="0.0.0",
                enabled=False,
                availability="UNAVAILABLE",
                input_schema={"asset_id": "str"},
                output_schema={"caption": "str"},
                resource_requirements={"min_ram_gb": 4.0},
                permitted_parameters={}
            )
        )

        # 4. Temporal Change Detection (Activated in Prompt 5)
        self.register_tool(
            ToolDefinition(
                name="adapter_temporal_change",
                tool_id="adapter_temporal_change",
                task_types=[TaskFamily.TEMPORAL_CHANGE],
                modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
                model_name="HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff",
                version="1.0.0",
                enabled=True,
                availability="AVAILABLE",
                input_schema={"t1_asset_id": "str", "t2_asset_id": "str"},
                output_schema={"change_clusters": "List[TemporalChangeCluster]", "changed_area_ha": "float", "change_verdict": "str"},
                resource_requirements={"min_ram_gb": 1.0, "accelerator": "mps/cuda/cpu"},
                permitted_parameters={"threshold": {"type": "float", "min": 0.01, "max": 1.0}, "method": {"type": "str", "options": ["siamunet_diff", "analytical_cva"]}}
            )
        )

        # 5. Temporal Change VQA (Activated in Prompt 5)
        self.register_tool(
            ToolDefinition(
                name="adapter_temporal_change_vqa",
                tool_id="adapter_temporal_change_vqa",
                task_types=[TaskFamily.TEMPORAL_CHANGE_VQA],
                modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
                model_name="satquery/temporal-change-vqa-engine",
                version="1.0.0",
                enabled=True,
                availability="AVAILABLE",
                input_schema={"t1_asset_id": "str", "t2_asset_id": "str", "question": "str"},
                output_schema={"answer": "str", "change_verdict": "str", "changed_area_ha": "float"},
                resource_requirements={"min_ram_gb": 1.0},
                permitted_parameters={"threshold": {"type": "float", "min": 0.01, "max": 1.0}}
            )
        )

        # 6. Optical + SAR Fusion (Activated in Prompt 8)
        self.register_tool(
            ToolDefinition(
                name="adapter_optical_sar_fusion",
                tool_id="adapter_optical_sar_fusion",
                task_types=[TaskFamily.OPTICAL_SAR_ANALYSIS],
                modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR],
                model_name="SatQuery-OpticalSAR-FusionAdapter-v1",
                version="1.0.0",
                enabled=True,
                availability="AVAILABLE",
                input_schema={"optical_asset_id": "str", "sar_asset_id": "str", "query": "str"},
                output_schema={"agreement_regions": "List[Dict]", "disagreement_regions": "List[Dict]", "uncertainty_state": "str"},
                resource_requirements={"min_ram_gb": 2.0},
                permitted_parameters={"sar_db_threshold": {"type": "float", "min": -35.0, "max": 0.0}}
            )
        )

    def register_tool(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        if tool.tool_id:
            self._tools[tool.tool_id] = tool
        logger.info(f"Registered tool: {tool.name} (availability: {tool.availability})")

    def register_adapter(self, adapter: Any) -> None:
        self._adapters[adapter.task_family] = adapter
        tool_id = f"adapter_{adapter.task_family.value.lower()}"
        tool_def = ToolDefinition(
            name=tool_id,
            tool_id=tool_id,
            task_types=[adapter.task_family],
            modalities=adapter.supported_modalities,
            model_name=adapter.model_name,
            version="1.0.0",
            enabled=True,
            availability="AVAILABLE",
            callable_ref=adapter
        )
        self.register_tool(tool_def)
        logger.info(f"Registered ModelAdapter for {adapter.task_family.value}: {adapter.model_name}")

    def get_adapter(self, task: TaskFamily) -> Optional[Any]:
        if task in self._adapters:
            return self._adapters[task]
            
        # Lazy load default specialist adapters on first demand
        if task == TaskFamily.SINGLE_VQA:
            from app.services.models.vqa_adapter import VqaAdapter
            adapter = VqaAdapter()
            self.register_adapter(adapter)
            return adapter
        elif task == TaskFamily.SINGLE_GROUNDING:
            from app.services.models.grounding_adapter import GroundingAdapter
            adapter = GroundingAdapter()
            self.register_adapter(adapter)
            return adapter
        elif task == TaskFamily.TEMPORAL_CHANGE:
            from app.services.models.temporal_change import TemporalChangeAdapter
            adapter = TemporalChangeAdapter()
            self.register_adapter(adapter)
            return adapter
        elif task == TaskFamily.TEMPORAL_CHANGE_VQA:
            from app.services.models.temporal_vqa import TemporalVqaAdapter
            adapter = TemporalVqaAdapter()
            self.register_adapter(adapter)
            return adapter
        elif task == TaskFamily.OPTICAL_SAR_ANALYSIS:
            from app.services.models.optical_sar_fusion import optical_sar_fusion_adapter
            self.register_adapter(optical_sar_fusion_adapter)
            return optical_sar_fusion_adapter

        return None

    def unload_adapter(self, task: TaskFamily) -> None:
        """Explicitly unload an adapter to free memory on constrained hosts."""
        if task in self._adapters:
            adapter = self._adapters[task]
            if hasattr(adapter, "unload"):
                adapter.unload()
            logger.info(f"Unloaded adapter for task: {task.value}")

    def unload_all_adapters(self) -> None:
        """Unload all active adapters."""
        for task in list(self._adapters.keys()):
            self.unload_adapter(task)

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def get_tool_by_id(self, tool_id: str) -> Optional[ToolDefinition]:
        return self._tools.get(tool_id)

    def list_tools(self, enabled_only: bool = True) -> List[ToolDefinition]:
        # Return unique tool definitions (since tool_id and name may both be keyed)
        unique_tools = list({t.name: t for t in self._tools.values()}.values())
        if enabled_only:
            return [t for t in unique_tools if t.enabled]
        return unique_tools

    def get_tools_for_task(self, task: TaskFamily, modality: Optional[Modality] = None) -> List[ToolDefinition]:
        matched = []
        for t in self.list_tools(enabled_only=False):
            if task in t.task_types:
                if modality is None or modality in t.modalities:
                    matched.append(t)
        return matched

    def has_tool_for_task(self, task: TaskFamily, modality: Optional[Modality] = None) -> bool:
        available = [t for t in self.get_tools_for_task(task, modality) if t.enabled and t.availability == "AVAILABLE"]
        return len(available) > 0


# Global registry singleton
model_tool_registry = ModelToolRegistry()
model_tool_registry._register_default_tool_metadata()

