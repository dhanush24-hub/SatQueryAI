from app.api.routes.health import router as health_router
from app.api.routes.uploads import router as uploads_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.jobs import router as jobs_router

__all__ = ["health_router", "uploads_router", "analysis_router", "jobs_router"]
