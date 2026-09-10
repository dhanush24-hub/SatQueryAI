from app.domain.modalities import Modality
from app.domain.tasks import TaskFamily
from app.registry.model_registry import ModelToolRegistry, ToolDefinition


def test_model_tool_registry():
    registry = ModelToolRegistry()

    # Zero tools initially
    assert len(registry.list_tools()) == 0
    assert not registry.has_tool_for_task(TaskFamily.SINGLE_VQA)

    # Register a tool
    vqa_tool = ToolDefinition(
        name="optical_vqa_tool",
        task_types=[TaskFamily.SINGLE_VQA, TaskFamily.SINGLE_CAPTION],
        modalities=[Modality.OPTICAL, Modality.MULTISPECTRAL],
        model_name="geochat-7b",
        version="1.0.0",
        enabled=True
    )
    registry.register_tool(vqa_tool)

    assert len(registry.list_tools()) == 1
    assert registry.get_tool("optical_vqa_tool") == vqa_tool
    assert registry.has_tool_for_task(TaskFamily.SINGLE_VQA)
    assert registry.has_tool_for_task(TaskFamily.SINGLE_VQA, Modality.OPTICAL)
    assert not registry.has_tool_for_task(TaskFamily.SINGLE_VQA, Modality.SAR)
    assert not registry.has_tool_for_task(TaskFamily.TEMPORAL_CHANGE)

    # Disabling tool
    vqa_tool.enabled = False
    assert len(registry.list_tools(enabled_only=True)) == 0
    assert len(registry.list_tools(enabled_only=False)) == 1
    assert not registry.has_tool_for_task(TaskFamily.SINGLE_VQA)
