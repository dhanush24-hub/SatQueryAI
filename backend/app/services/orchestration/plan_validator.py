from typing import Dict, List, Optional, Tuple
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.domain.modalities import Modality
from app.schemas.image_asset import ImageAsset
from app.schemas.compatibility import PairType
from app.registry.model_registry import model_tool_registry
from app.services.orchestration.plan_schema import WorkflowPlan, PlanStep
from app.services.geospatial.compatibility import check_pair_compatibility
from app.core.logging import logger


class PlanValidator:
    """
    Validates workflow plans prior to model execution.
    Enforces server-side safety invariants: asset existence, modality support,
    tool availability, parameter bounds, pair compatibility, and acyclic step dependencies.
    """

    def validate_plan(
        self,
        plan: WorkflowPlan,
        assets: List[ImageAsset]
    ) -> Tuple[bool, Optional[str], AnalysisStatus]:
        """
        Validates the plan against input assets and registry capabilities.
        Returns (is_valid, error_reason, failure_status).
        """
        # 1. Asset existence and count checks
        if not assets:
            return False, "No valid image assets provided for execution.", AnalysisStatus.VALIDATION_FAILED

        expected_count = 1 if plan.task_family in [TaskFamily.SINGLE_VQA, TaskFamily.SINGLE_GROUNDING, TaskFamily.SINGLE_CAPTION] else 2
        if len(assets) < expected_count:
            return False, f"Task {plan.task_family.value} requires {expected_count} image(s), but {len(assets)} was provided.", AnalysisStatus.VALIDATION_FAILED

        # Domain suitability check: spatial dimensions must provide sufficient context
        for asset in assets:
            if asset.width and asset.height and (asset.width < 32 or asset.height < 32):
                return (
                    False,
                    f"Asset '{asset.id}' dimensions {asset.width}x{asset.height} are out of domain / too small for spatial intelligence.",
                    AnalysisStatus.INSUFFICIENT_EVIDENCE
                )

        # 2. Step dependencies and acyclic DAG check
        step_ids = {s.step_id for s in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    return False, f"Step '{step.step_id}' references non-existent dependency '{dep}'.", AnalysisStatus.VALIDATION_FAILED

        if self._has_cycles(plan.steps):
            return False, "Plan contains cyclic step dependencies.", AnalysisStatus.VALIDATION_FAILED

        # 3. Tool availability and capability checks
        for step in plan.steps:
            tool = model_tool_registry.get_tool(step.tool_id)
            if tool is None:
                return False, f"Referenced tool '{step.tool_id}' is not registered in ModelToolRegistry.", AnalysisStatus.MODEL_UNAVAILABLE

            if not tool.enabled or tool.availability != "AVAILABLE":
                return (
                    False,
                    f"Tool '{step.tool_id}' for task '{step.task_family.value}' is currently UNAVAILABLE. "
                    "This capability is not yet implemented or active in this prompt slice.",
                    AnalysisStatus.MODEL_UNAVAILABLE
                )

            # Modality compatibility check
            for asset in assets:
                if asset.modality not in tool.modalities:
                    if asset.modality == Modality.SAR:
                        return (
                            False,
                            f"SAR imagery is not supported by optical specialist model '{tool.name}'. "
                            "Dedicated SAR specialist model is required per SIH PS 26167 guidelines.",
                            AnalysisStatus.MODEL_UNAVAILABLE
                        )
                    return (
                        False,
                        f"Asset '{asset.id}' modality '{asset.modality.value}' is not supported by tool '{tool.name}'. "
                        f"Supported modalities: {[m.value for m in tool.modalities]}.",
                        AnalysisStatus.MODEL_UNAVAILABLE
                    )

            # Parameter bounds check
            param_err = self._validate_parameters(step, tool.permitted_parameters)
            if param_err:
                return False, param_err, AnalysisStatus.VALIDATION_FAILED

        # 4. Multi-image pair compatibility check
        if len(assets) >= 2:
            pair_type = PairType.OPTICAL_SAR if plan.task_family == TaskFamily.OPTICAL_SAR_ANALYSIS else PairType.TEMPORAL
            compat_result = check_pair_compatibility(assets[0], assets[1], pair_type=pair_type)
            if not compat_result.compatible:
                return (
                    False,
                    f"Paired images are incompatible: {'; '.join(compat_result.warnings)}",
                    AnalysisStatus.VALIDATION_FAILED
                )

        logger.info(f"Plan '{plan.plan_id}' passed all safety and capability validations.")
        return True, None, AnalysisStatus.COMPLETED

    def _validate_parameters(self, step: PlanStep, permitted: Dict) -> Optional[str]:
        for p_name, p_val in step.parameters.items():
            if p_name in permitted:
                bounds = permitted[p_name]
                p_type = bounds.get("type")
                if p_type == "float" and isinstance(p_val, (int, float)):
                    if "min" in bounds and p_val < bounds["min"]:
                        return f"Parameter '{p_name}' value {p_val} is below allowed minimum {bounds['min']}."
                    if "max" in bounds and p_val > bounds["max"]:
                        return f"Parameter '{p_name}' value {p_val} is above allowed maximum {bounds['max']}."
                elif p_type == "int" and isinstance(p_val, int):
                    if "min" in bounds and p_val < bounds["min"]:
                        return f"Parameter '{p_name}' value {p_val} is below allowed minimum {bounds['min']}."
                    if "max" in bounds and p_val > bounds["max"]:
                        return f"Parameter '{p_name}' value {p_val} is above allowed maximum {bounds['max']}."
        return None

    def _has_cycles(self, steps: List[PlanStep]) -> bool:
        """Topological cycle detection for execution DAG."""
        adj = {s.step_id: list(s.dependencies) for s in steps}
        visited: Dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited

        def dfs(node: str) -> bool:
            visited[node] = 1
            for neighbor in adj.get(node, []):
                if visited.get(neighbor, 0) == 1:
                    return True
                if visited.get(neighbor, 0) == 0 and dfs(neighbor):
                    return True
            visited[node] = 2
            return False

        for s in steps:
            if visited.get(s.step_id, 0) == 0:
                if dfs(s.step_id):
                    return True
        return False
