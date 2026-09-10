import uuid
from typing import Any, Dict, List
from app.domain.tasks import TaskFamily
from app.schemas.image_asset import ImageAsset
from app.registry.model_registry import model_tool_registry
from app.services.orchestration.plan_schema import WorkflowPlan, PlanStep
from app.core.logging import logger


class WorkflowPlanner:
    """
    Constructs bounded, typed workflow execution DAGs from classified tasks
    and registered tool definitions. Prevents arbitrary tool invention or unsafe execution.
    """

    def create_plan(
        self,
        task_family: TaskFamily,
        query: str,
        assets: List[ImageAsset],
        classification_meta: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> WorkflowPlan:
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        asset_ids = [a.id for a in assets]
        primary_asset_id = asset_ids[0] if asset_ids else ""

        steps: List[PlanStep] = []
        deps: Dict[str, List[str]] = {}

        # Check if composite query requires both VQA and Grounding
        is_composite = classification_meta.get("is_composite", False)

        if task_family == TaskFamily.SINGLE_VQA and not is_composite:
            steps.append(
                PlanStep(
                    step_id="step_vqa_1",
                    tool_id="adapter_single_vqa",
                    task_family=TaskFamily.SINGLE_VQA,
                    description="Execute single-image visual question answering via BLIP-VQA",
                    input_bindings={"asset_id": primary_asset_id, "question": query},
                    parameters={"max_new_tokens": parameters.get("max_new_tokens", 32)},
                    dependencies=[],
                    timeout_seconds=45.0
                )
            )
            deps["step_vqa_1"] = []

        elif task_family == TaskFamily.SINGLE_GROUNDING and not is_composite:
            # Extract query phrases for grounding
            query_phrases = self._extract_grounding_phrases(query)
            threshold = float(parameters.get("threshold", 0.08))

            steps.append(
                PlanStep(
                    step_id="step_grounding_1",
                    tool_id="adapter_single_grounding",
                    task_family=TaskFamily.SINGLE_GROUNDING,
                    description="Execute open-vocabulary text-guided grounding via OWL-ViT",
                    input_bindings={"asset_id": primary_asset_id, "queries": query_phrases},
                    parameters={"threshold": threshold},
                    dependencies=[],
                    timeout_seconds=45.0
                )
            )
            deps["step_grounding_1"] = []

        elif is_composite:
            # Composite Plan: Sequenced VQA and Grounding
            steps.append(
                PlanStep(
                    step_id="step_vqa_1",
                    tool_id="adapter_single_vqa",
                    task_family=TaskFamily.SINGLE_VQA,
                    description="Execute single-image visual question answering for descriptive inquiry",
                    input_bindings={"asset_id": primary_asset_id, "question": query},
                    parameters={"max_new_tokens": parameters.get("max_new_tokens", 32)},
                    dependencies=[],
                    timeout_seconds=45.0
                )
            )
            deps["step_vqa_1"] = []

            query_phrases = self._extract_grounding_phrases(query)
            threshold = float(parameters.get("threshold", 0.08))

            steps.append(
                PlanStep(
                    step_id="step_grounding_2",
                    tool_id="adapter_single_grounding",
                    task_family=TaskFamily.SINGLE_GROUNDING,
                    description="Execute text-guided grounding to localize identified features",
                    input_bindings={"asset_id": primary_asset_id, "queries": query_phrases},
                    parameters={"threshold": threshold},
                    dependencies=["step_vqa_1"],
                    timeout_seconds=45.0
                )
            )
            deps["step_grounding_2"] = ["step_vqa_1"]

        elif task_family == TaskFamily.SINGLE_CAPTION:
            steps.append(
                PlanStep(
                    step_id="step_caption_1",
                    tool_id="adapter_single_caption",
                    task_family=TaskFamily.SINGLE_CAPTION,
                    description="Generate dense remote sensing scene description",
                    input_bindings={"asset_id": primary_asset_id},
                    parameters={},
                    dependencies=[],
                    timeout_seconds=45.0
                )
            )
            deps["step_caption_1"] = []

        elif task_family in [TaskFamily.TEMPORAL_CHANGE, TaskFamily.TEMPORAL_CHANGE_VQA]:
            tool_id = "adapter_temporal_change_vqa" if task_family == TaskFamily.TEMPORAL_CHANGE_VQA else "adapter_temporal_change"
            steps.append(
                PlanStep(
                    step_id="step_temporal_1",
                    tool_id=tool_id,
                    task_family=task_family,
                    description=f"Execute bi-temporal change analysis ({task_family.value})",
                    input_bindings={
                        "t1_asset_id": asset_ids[0] if len(asset_ids) > 0 else "",
                        "t2_asset_id": asset_ids[1] if len(asset_ids) > 1 else "",
                        "query": query
                    },
                    parameters=dict(parameters),
                    dependencies=[],
                    timeout_seconds=60.0
                )
            )
            deps["step_temporal_1"] = []

        elif task_family == TaskFamily.OPTICAL_SAR_ANALYSIS:
            steps.append(
                PlanStep(
                    step_id="step_fusion_1",
                    tool_id="adapter_optical_sar_fusion",
                    task_family=TaskFamily.OPTICAL_SAR_ANALYSIS,
                    description="Execute cross-modal Optical + SAR feature fusion and analysis",
                    input_bindings={
                        "optical_asset_id": asset_ids[0] if len(asset_ids) > 0 else "",
                        "sar_asset_id": asset_ids[1] if len(asset_ids) > 1 else "",
                        "query": query
                    },
                    parameters={},
                    dependencies=[],
                    timeout_seconds=60.0
                )
            )
            deps["step_fusion_1"] = []

        plan = WorkflowPlan(
            plan_id=plan_id,
            task_family=task_family,
            input_asset_ids=asset_ids,
            steps=steps,
            dependencies=deps,
            permitted_parameters={
                "threshold": {"type": "float", "min": 0.001, "max": 1.0},
                "max_new_tokens": {"type": "int", "min": 1, "max": 64},
                "method": {"type": "str", "options": ["siamunet_diff", "analytical_cva"]},
                "user_order": {"type": "list"}
            },
            expected_outputs=["answer", "evidence", "execution_summary"],
            classification_meta=classification_meta,
            validation_status="PENDING"
        )
        logger.info(f"Constructed plan {plan_id} for {task_family.value} with {len(steps)} step(s).")
        return plan

    def _extract_grounding_phrases(self, query: str) -> List[str]:
        """Extract clean target phrases for grounding from natural language instructions."""
        clean = query.strip()
        # Remove common instruction preambles
        prefixes = [
            r"^(highlight|locate|find|where\s+is|where\s+are|point\s+out|show\s+me|box|outline)\s+(the\s+|all\s+)?",
            r"^(can\s+you\s+(highlight|locate|find|identify))\s+(the\s+|all\s+)?",
        ]
        import re
        for p in prefixes:
            clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()

        # Remove trailing punctuation
        clean = clean.rstrip(".?!")
        return [clean] if clean else ["target object"]
