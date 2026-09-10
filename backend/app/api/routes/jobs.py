from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.domain.tasks import AnalysisStatus

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


class JobStatusResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    progress: float
    message: Optional[str] = None
    result_id: Optional[str] = None


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Check asynchronous job execution status.
    In this foundation stage, any queried job returns a clean state or 404.
    """
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Job '{job_id}' not found. Asynchronous queuing pipeline will be wired in model integration."
    )
