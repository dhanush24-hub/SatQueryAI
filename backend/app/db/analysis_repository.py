import json
from typing import List, Optional
from app.db.session import get_db_connection
from app.schemas.analysis import AnalysisResult, AnalysisRequest, AnalysisHistorySummary
from app.domain.tasks import TaskFamily, AnalysisStatus
from app.core.logging import logger


class AnalysisRepository:
    """
    Persistent SQLite storage repository for completed and historical analysis results.
    Ensures reproducibility and auditability without re-running model inference.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path

    def _conn(self):
        return get_db_connection(self._db_path)

    def save_analysis(
        self,
        result: AnalysisResult,
        request: AnalysisRequest,
        report_pdf_path: Optional[str] = None
    ) -> None:
        """Persists a complete AnalysisResult into SQLite."""
        task_str = result.task.value if hasattr(result.task, "value") else str(result.task or "UNKNOWN")
        status_str = result.status.value if hasattr(result.status, "value") else str(result.status or "COMPLETED")
        image_ids_json = json.dumps(request.image_ids)
        result_json = result.model_dump_json()

        query = """
        INSERT OR REPLACE INTO analysis_results (
            id, query, task, status, answer, confidence,
            evidence_state, image_ids_json, result_json,
            report_pdf_path, created_at
        ) VALUES (
            :id, :query, :task, :status, :answer, :confidence,
            :evidence_state, :image_ids_json, :result_json,
            :report_pdf_path, datetime('now')
        );
        """
        conn = self._conn()
        with conn:
            conn.execute(query, {
                "id": result.id,
                "query": request.query,
                "task": task_str,
                "status": status_str,
                "answer": result.answer,
                "confidence": result.confidence,
                "evidence_state": result.evidence_state,
                "image_ids_json": image_ids_json,
                "result_json": result_json,
                "report_pdf_path": report_pdf_path
            })
        logger.info(f"Persisted analysis result '{result.id}' ({task_str}) to SQLite.")

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Loads and deserializes an exact historical analysis result by ID."""
        conn = self._conn()
        cur = conn.execute("SELECT result_json FROM analysis_results WHERE id = ?", (analysis_id,))
        row = cur.fetchone()
        if not row:
            return None
        data = json.loads(row["result_json"])
        return AnalysisResult.model_validate(data)

    def list_analyses(self, limit: int = 50) -> List[AnalysisHistorySummary]:
        """Lists recent analyses for sidebar/history view with thumbnails and summaries."""
        conn = self._conn()
        cur = conn.execute("""
            SELECT id, query, task, status, answer, evidence_state, image_ids_json, created_at
            FROM analysis_results
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()

        summaries = []
        for r in rows:
            try:
                img_ids = json.loads(r["image_ids_json"])
            except Exception:
                img_ids = []

            # Thumbnail from first image asset if available
            thumbnail = f"/api/uploads/previews/{img_ids[0]}.png" if img_ids else None

            # Safe enum parsing
            try:
                task_enum = TaskFamily(r["task"])
            except Exception:
                task_enum = None

            try:
                status_enum = AnalysisStatus(r["status"])
            except Exception:
                status_enum = AnalysisStatus.COMPLETED

            summaries.append(AnalysisHistorySummary(
                id=r["id"],
                query=r["query"],
                timestamp=r["created_at"],
                image_ids=img_ids,
                task=task_enum,
                status=status_enum,
                thumbnail=thumbnail,
                summary=r["answer"][:120] + ("..." if len(r["answer"]) > 120 else ""),
                evidence_state=r["evidence_state"] or "SUPPORTED"
            ))
        return summaries


analysis_repository = AnalysisRepository()
