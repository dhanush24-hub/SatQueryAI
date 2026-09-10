import time
from typing import Any, Dict, List, Optional, Tuple
from app.domain.tasks import TaskFamily
from app.schemas.execution import ExecutionStep
from app.schemas.image_asset import ImageAsset
from app.registry.model_registry import model_tool_registry
from app.services.orchestration.plan_schema import WorkflowPlan, PlanStep
from app.services.models.schemas import VqaRequest, GroundingRequest
from app.core.logging import logger


class WorkflowExecutor:
    """
    Dependency-aware execution engine for workflow plans.
    Enforces host memory safety (preventing concurrent model loads on 8 GB RAM),
    tracks timing telemetry, manages step lifecycles, and captures observable traces.
    """

    def execute_plan(
        self,
        plan: WorkflowPlan,
        assets: List[ImageAsset]
    ) -> Tuple[Dict[str, Any], List[ExecutionStep], Optional[str]]:
        """
        Executes steps in the workflow plan.
        Returns: (step_outputs, execution_steps, error_message)
        """
        step_outputs: Dict[str, Any] = {}
        execution_trace: List[ExecutionStep] = []
        error_msg: Optional[str] = None

        logger.info(f"Starting execution of plan '{plan.plan_id}' with {len(plan.steps)} steps.")

        for step in plan.steps:
            # Check dependencies
            missing_dep = [dep for dep in step.dependencies if dep not in step_outputs]
            if missing_dep:
                step.status = "blocked"
                execution_trace.append(
                    ExecutionStep(
                        step=step.description,
                        tool_name=step.tool_id,
                        status="blocked",
                        summary=f"Blocked due to prerequisite step failures: {missing_dep}"
                    )
                )
                continue

            step.status = "running"
            t0 = time.time()

            try:
                # Host Memory Optimization: Unload other adapters if running a heavy vision model
                self._manage_host_memory(step.task_family)

                output = self._execute_step(step, assets)
                duration_ms = int((time.time() - t0) * 1000)
                step.status = "completed"
                step_outputs[step.step_id] = output

                # Create observable step trace record
                tool = model_tool_registry.get_tool(step.tool_id)
                adapter = model_tool_registry.get_adapter(step.task_family)
                adapter_model = getattr(adapter, "model_name", None)
                model_name = getattr(output, "model_name", adapter_model or (tool.model_name if tool else None))
                tool_name = adapter_model or step.tool_id

                summary = self._generate_step_summary(step, output, duration_ms)
                execution_trace.append(
                    ExecutionStep(
                        step=step.description,
                        tool_name=tool_name,
                        model_name=model_name,
                        parameters=step.parameters,
                        status="completed",
                        duration_ms=duration_ms,
                        summary=summary
                    )
                )

            except Exception as e:
                duration_ms = int((time.time() - t0) * 1000)
                step.status = "failed"
                error_msg = f"Step '{step.step_id}' failed: {str(e)}"
                logger.error(f"Execution failure in step {step.step_id}: {e}")

                execution_trace.append(
                    ExecutionStep(
                        step=step.description,
                        tool_name=step.tool_id,
                        status="failed",
                        duration_ms=duration_ms,
                        summary=f"Failed with error: {str(e)}"
                    )
                )
                # Halt execution on step failure (no unbounded loop)
                break

        return step_outputs, execution_trace, error_msg

    def _execute_step(self, step: PlanStep, assets: List[ImageAsset]) -> Any:
        adapter = model_tool_registry.get_adapter(step.task_family)
        if adapter is None:
            raise ValueError(f"No executable adapter found for task {step.task_family.value}.")

        bindings = step.input_bindings
        asset_id = bindings.get("asset_id", assets[0].id if assets else "")

        if step.task_family == TaskFamily.SINGLE_VQA:
            req = VqaRequest(
                asset_id=asset_id,
                question=bindings.get("question", "")
            )
            return adapter.predict(req)

        elif step.task_family == TaskFamily.SINGLE_GROUNDING:
            queries = bindings.get("queries", ["target object"])
            threshold = float(step.parameters.get("threshold", 0.08))
            req = GroundingRequest(
                asset_id=asset_id,
                queries=queries,
                threshold=threshold
            )
            return adapter.predict(req)

        elif step.task_family == TaskFamily.TEMPORAL_CHANGE:
            from app.services.models.schemas import TemporalChangeRequest
            t1_id = bindings.get("t1_asset_id", assets[0].id if len(assets) > 0 else "")
            t2_id = bindings.get("t2_asset_id", assets[1].id if len(assets) > 1 else "")
            req = TemporalChangeRequest(
                t1_asset_id=t1_id,
                t2_asset_id=t2_id,
                threshold=float(step.parameters.get("threshold", 0.50)),
                method=str(step.parameters.get("method", "siamunet_diff")),
                parameters=step.parameters
            )
            return adapter.predict(req)

        elif step.task_family == TaskFamily.TEMPORAL_CHANGE_VQA:
            from app.services.models.schemas import TemporalVqaRequest
            t1_id = bindings.get("t1_asset_id", assets[0].id if len(assets) > 0 else "")
            t2_id = bindings.get("t2_asset_id", assets[1].id if len(assets) > 1 else "")
            req = TemporalVqaRequest(
                t1_asset_id=t1_id,
                t2_asset_id=t2_id,
                question=bindings.get("query", ""),
                parameters=step.parameters
            )
            return adapter.predict(req)

        elif step.task_family == TaskFamily.OPTICAL_SAR_ANALYSIS:
            from app.services.models.optical_sar_fusion import OpticalSarFusionRequest
            opt_id = bindings.get("optical_asset_id", assets[0].id if len(assets) > 0 else "")
            sar_id = bindings.get("sar_asset_id", assets[1].id if len(assets) > 1 else "")
            req = OpticalSarFusionRequest(
                optical_asset_id=opt_id,
                sar_asset_id=sar_id,
                query=bindings.get("query", ""),
                parameters=step.parameters
            )
            return adapter.predict(req)

        raise NotImplementedError(f"Task family {step.task_family.value} execution not implemented.")

    def _manage_host_memory(self, target_task: TaskFamily) -> None:
        """
        Unload non-target heavy models to prevent concurrent resident memory exhaustion
        on 8 GB Apple Silicon host.
        """
        if target_task == TaskFamily.SINGLE_VQA:
            model_tool_registry.unload_adapter(TaskFamily.SINGLE_GROUNDING)
            model_tool_registry.unload_adapter(TaskFamily.TEMPORAL_CHANGE)
            model_tool_registry.unload_adapter(TaskFamily.OPTICAL_SAR_ANALYSIS)
        elif target_task == TaskFamily.SINGLE_GROUNDING:
            model_tool_registry.unload_adapter(TaskFamily.SINGLE_VQA)
            model_tool_registry.unload_adapter(TaskFamily.TEMPORAL_CHANGE)
            model_tool_registry.unload_adapter(TaskFamily.OPTICAL_SAR_ANALYSIS)
        elif target_task in [TaskFamily.TEMPORAL_CHANGE, TaskFamily.TEMPORAL_CHANGE_VQA]:
            model_tool_registry.unload_adapter(TaskFamily.SINGLE_VQA)
            model_tool_registry.unload_adapter(TaskFamily.SINGLE_GROUNDING)
            model_tool_registry.unload_adapter(TaskFamily.OPTICAL_SAR_ANALYSIS)
        elif target_task == TaskFamily.OPTICAL_SAR_ANALYSIS:
            model_tool_registry.unload_adapter(TaskFamily.SINGLE_VQA)
            model_tool_registry.unload_adapter(TaskFamily.SINGLE_GROUNDING)
            model_tool_registry.unload_adapter(TaskFamily.TEMPORAL_CHANGE)

    def _generate_step_summary(self, step: PlanStep, output: Any, duration_ms: int) -> str:
        if step.task_family == TaskFamily.SINGLE_VQA:
            ans = getattr(output, "answer", "No answer")
            dev = getattr(output, "device", "unknown")
            return f"VQA completed on {dev} in {duration_ms}ms. Predicted: '{ans}'"
        elif step.task_family == TaskFamily.SINGLE_GROUNDING:
            boxes = getattr(output, "boxes", [])
            dev = getattr(output, "device", "unknown")
            return f"Grounding completed on {dev} in {duration_ms}ms. Detected {len(boxes)} raw candidate region(s)."
        elif step.task_family == TaskFamily.TEMPORAL_CHANGE:
            clusters = getattr(output, "change_clusters", [])
            area_ha = getattr(output, "changed_area_ha", None)
            verdict = getattr(output, "change_verdict", "UNKNOWN")
            area_str = f"{area_ha:.2f}ha" if area_ha is not None else "unreferenced"
            return f"Temporal change detection completed in {duration_ms}ms: verdict='{verdict}', area={area_str} across {len(clusters)} cluster(s)."
        elif step.task_family == TaskFamily.TEMPORAL_CHANGE_VQA:
            verdict = getattr(output, "change_verdict", "UNKNOWN")
            ans = getattr(output, "answer", "")[:60]
            return f"Temporal VQA completed in {duration_ms}ms: verdict='{verdict}'. Answer: '{ans}...'"
        elif step.task_family == TaskFamily.OPTICAL_SAR_ANALYSIS:
            state = getattr(output, "uncertainty_state", "UNKNOWN")
            agree_ratio = getattr(output, "agreement_ratio_pct", 0.0)
            return f"Optical+SAR cross-modal fusion completed in {duration_ms}ms: state='{state}', mutual water agreement={agree_ratio}%."
        return f"Step completed in {duration_ms}ms."
