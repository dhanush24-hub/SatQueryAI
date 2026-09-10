import json
import os
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Response
from fastapi.responses import FileResponse
from app.schemas.analysis import AnalysisRequest, AnalysisResult, AnalysisHistorySummary
from app.db.asset_repository import asset_repository
from app.db.analysis_repository import analysis_repository
from app.services.orchestration.controller import workflow_controller
from app.services.reporting.pdf_generator import pdf_generator
from app.services.storage.local import storage_service
from app.core.logging import logger

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

# Fallback in-memory storage for non-persisted test mocking
_analysis_store: Dict[str, AnalysisResult] = {}


@router.post("", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
async def request_analysis(request: AnalysisRequest) -> AnalysisResult:
    """
    Execute agentic remote sensing analysis.
    Uses WorkflowController to interpret intent, construct a validated plan DAG,
    execute specialist models from ModelToolRegistry, assess evidence quality,
    generate downloadable report PDF, and persist result to SQLite.
    """
    logger.info(f"Received AnalysisRequest: query='{request.query}', image_ids={request.image_ids}")

    if not request.image_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image asset ID is required."
        )

    # Validate asset existence upfront for client 404 semantics
    assets = []
    for asset_id in request.image_ids:
        asset = asset_repository.get_asset(asset_id)
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image asset '{asset_id}' not found in registry."
            )
        assets.append(asset)

    result = workflow_controller.execute_analysis(request)

    # Generate PDF report
    pdf_filename = None
    try:
        pdf_filename = pdf_generator.generate_report(result, assets)
    except Exception as e:
        logger.warning(f"Failed to generate PDF report for {result.id}: {e}")

    # Persist to SQLite
    try:
        analysis_repository.save_analysis(result, request, pdf_filename)
    except Exception as e:
        logger.warning(f"Failed to persist analysis {result.id} to SQLite: {e}")

    _analysis_store[result.id] = result
    return result


@router.get("", response_model=List[AnalysisHistorySummary])
async def list_analyses(limit: int = Query(default=50, ge=1, le=200)) -> List[AnalysisHistorySummary]:
    """
    List persistent analysis history for workspace sidebar.
    """
    try:
        return analysis_repository.list_analyses(limit=limit)
    except Exception as e:
        logger.warning(f"Could not load analysis history from SQLite: {e}")
        return []


@router.get("/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(analysis_id: str) -> AnalysisResult:
    """
    Retrieve existing persistent analysis result by ID without re-running models.
    """
    # 1. Check persistent SQLite repository
    result = None
    try:
        result = analysis_repository.get_analysis(analysis_id)
    except Exception as e:
        logger.warning(f"Error querying SQLite for {analysis_id}: {e}")

    # 2. Fallback to in-memory store
    if not result:
        result = _analysis_store.get(analysis_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis result '{analysis_id}' not found."
        )
    return result


@router.get("/{analysis_id}/report.pdf")
async def download_pdf_report(analysis_id: str):
    """
    Download publication-quality PDF report for completed analysis.
    """
    result = await get_analysis(analysis_id)
    pdf_filename = f"satquery_report_{analysis_id}.pdf"
    pdf_path = storage_service.get_full_path(pdf_filename)

    if not os.path.exists(pdf_path):
        # Regenerate on demand if missing
        assets = []
        for ev in result.evidence:
            if ev.image_id:
                a = asset_repository.get_asset(ev.image_id)
                if a and a not in assets:
                    assets.append(a)
        pdf_generator.generate_report(result, assets, output_filename=pdf_filename)

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate or locate PDF report."
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"SatQuery_Report_{analysis_id}.pdf"
    )


@router.get("/{analysis_id}/export.json")
async def export_result_json(analysis_id: str):
    """
    Export full result JSON for machine interoperability.
    """
    result = await get_analysis(analysis_id)
    json_bytes = result.model_dump_json(indent=2)
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=satquery_result_{analysis_id}.json"}
    )


@router.get("/{analysis_id}/export.geojson")
async def export_result_geojson(analysis_id: str):
    """
    Export authentic GeoJSON polygon features.
    Strictly forbidden (returns HTTP 400) if imagery lacks authentic georeferencing.
    """
    result = await get_analysis(analysis_id)

    # Check if authentic georeferencing exists
    if not (result.overlays and result.overlays.georeferenced):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GeoJSON export requires authentic georeferencing metadata (zero-fabrication policy). This imagery is unreferenced."
        )

    features = []
    for f in result.findings:
        if f.geometry:
            features.append({
                "type": "Feature",
                "geometry": f.geometry,
                "properties": {
                    "finding_id": f.finding_id,
                    "type": f.type,
                    "summary": f.summary,
                    "area_m2": f.area_m2,
                    "evidence_quality": f.evidence_quality,
                    "source_model": f.source_model,
                    "limitations": f.limitations
                }
            })

    if not features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No geographic polygon features found in this analysis result."
        )

    geojson_doc = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "analysis_id": analysis_id,
            "query": result.answer,
            "task": result.task.value if hasattr(result.task, "value") else str(result.task),
            "status": result.status.value if hasattr(result.status, "value") else str(result.status)
        }
    }

    return Response(
        content=json.dumps(geojson_doc, indent=2),
        media_type="application/geo+json",
        headers={"Content-Disposition": f"attachment; filename=satquery_{analysis_id}.geojson"}
    )
