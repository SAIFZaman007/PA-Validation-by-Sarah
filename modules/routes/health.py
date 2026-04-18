"""
Health-check route — GET /api/health

Used by load balancers, Docker HEALTHCHECK directives, and uptime monitors.
Returns HTTP 200 when the application and database are reachable.
Returns HTTP 503 when the database connection fails.
"""

from fastapi              import APIRouter
from sqlalchemy           import text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi              import Depends

from core.database import get_db
from modules.services.ml_pipeline import router as ml_router

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", summary="Application health check")
async def api_health(db: AsyncSession = Depends(get_db)):
    """
    Lightweight health-check endpoint.

    Checks:
      - Database reachability via a SELECT 1 query.
      - Whether the ML model was loaded successfully at startup.

    Returns HTTP 200 when everything is operational.
    Returns HTTP 503 when the database is unreachable.
    """
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    payload = {
        "status":   "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "unreachable",
        "model":    "loaded" if ml_router is not None else "not loaded",
    }

    from fastapi.responses import JSONResponse
    return JSONResponse(content=payload, status_code=200 if db_ok else 503)