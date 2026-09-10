import uuid
from typing import List
from app.domain.tasks import AnalysisStatus
from app.schemas.analysis import AnalysisRequest, AnalysisResult
from app.schemas.image_asset import ImageAsset
from app.db.asset_repository import asset_repository
from app.services.orchestration.task_classifier import TaskClassifier
from app.services.orchestration.planner import WorkflowPlanner
from app.services.orchestration.plan_validator import PlanValidator
from app.services.orchestration.executor import WorkflowExecutor
from app.services.orchestration.result_integrator import ResultIntegrator
from app.services.orchestration.plan_schema import EvidenceAssessment
from app.core.logging import logger


class WorkflowController:
    """
    Central agentic orchestrator for SatQuery AI.
    Interprets user queries and input imagery, selects specialist tools from ModelToolRegistry,
    validates execution plans, sequences tool execution, and synthesizes evidence-grounded findings.
    """

    def __init__(self):
        self.classifier = TaskClassifier()
        self.planner = WorkflowPlanner()
        self.validator = PlanValidator()
        self.executor = WorkflowExecutor()
        self.integrator = ResultIntegrator()

    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        logger.info(f"Initiating agentic workflow for query: '{request.query}' with {len(request.image_ids)} asset(s).")
        analysis_id = f"anl_{uuid.uuid4().hex[:12]}"

        # 1. Resolve and validate assets from persistent registry
        assets: List[ImageAsset] = []
        for asset_id in request.image_ids:
            asset = asset_repository.get_asset(asset_id)
            if not asset:
                return AnalysisResult(
                    id=analysis_id,
                    answer=f"Asset '{asset_id}' not found in registry.",
                    task=None,
                    confidence=None,
                    evidence=[],
                    warnings=[f"ImageAsset with ID '{asset_id}' does not exist or has expired."],
                    execution_summary=[],
                    status=AnalysisStatus.VALIDATION_FAILED,
                    evidence_assessment=EvidenceAssessment(
                        evidence_status="UNAVAILABLE",
                        raw_scores=[],
                        quality_flags=["ASSET_NOT_FOUND"],
                        limitations=["Missing input asset record"],
                        verification_method="none",
                        confidence=None
                    )
                )
            assets.append(asset)

        # 2. Query interpretation and task classification
        task_family, classification_meta = self.classifier.classify(request.query, assets)
        logger.info(f"Classified query into TaskFamily: {task_family.value} ({classification_meta.get('method')})")

        # 3. Plan construction
        parameters = request.parameters or {}
        plan = self.planner.create_plan(
            task_family=task_family,
            query=request.query,
            assets=assets,
            classification_meta=classification_meta,
            parameters=parameters
        )

        # 4. Server-side safety and capability validation
        is_valid, validation_err, failure_status = self.validator.validate_plan(plan, assets)
        if not is_valid:
            plan.validation_status = "BLOCKED"
            plan.rejection_reason = validation_err
            logger.warning(f"Plan '{plan.plan_id}' failed validation: {validation_err}")

            is_domain_err = any(k in (validation_err or "").lower() for k in ["domain", "too small"])
            ev_state = "OUT_OF_DOMAIN" if is_domain_err else "UNAVAILABLE"

            return AnalysisResult(
                id=analysis_id,
                answer=f"Workflow execution blocked: {validation_err}",
                task=task_family,
                confidence=None,
                evidence=[],
                warnings=[validation_err],
                execution_summary=[],
                status=failure_status,
                evidence_state=ev_state,
                plan_id=plan.plan_id,
                evidence_assessment=EvidenceAssessment(
                    evidence_status=ev_state,
                    raw_scores=[],
                    quality_flags=["VALIDATION_FAILED"],
                    limitations=[validation_err or "Server-side validation failed"],
                    verification_method="server_side_policy_audit",
                    confidence=None
                )
            )

        plan.validation_status = "VALID"

        # 5. Execute workflow plan through execution engine
        step_outputs, execution_trace, error_msg = self.executor.execute_plan(plan, assets)

        # 6. Synthesize evidence and assemble final result
        result = self.integrator.integrate(
            plan=plan,
            step_outputs=step_outputs,
            execution_trace=execution_trace,
            assets=assets,
            error_msg=error_msg,
            analysis_id=analysis_id
        )
        result.plan_id = plan.plan_id

        logger.info(f"Agentic workflow completed for {analysis_id}. Final status: {result.status.value}")
        return result


# Singleton workflow controller
workflow_controller = WorkflowController()
