"""
PA Validation System — FastAPI entry point.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib    import Path

import uvicorn
from fastapi                 import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# backend/ must be on sys.path so core/, modules/, src/ are importable
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

from core.config   import settings
from core.database import create_tables

from modules.routes.policies    import router as policies_router
from modules.routes.submissions import router as submissions_router
from modules.routes.health      import router as health_router

# Keep SQLAlchemy and watchfiles quiet in the console
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    await _warm_policy_cache()
    print("[startup] Database ready.")
    print(f"[startup] Model: {'loaded' if _model_loaded() else 'NOT loaded'}")
    yield
    print("[shutdown] Server shutting down.")


async def _warm_policy_cache() -> None:
    """Pre-load all stored policy rules into memory at startup."""
    from core.database import AsyncSessionLocal
    from core.models   import Policy
    from sqlalchemy    import select
    from modules.services.ml_pipeline import (
        extractor, clear_policy_rules, extend_policy_rules,
    )
    async with AsyncSessionLocal() as session:
        result  = await session.execute(select(Policy))
        records = result.scalars().all()
        clear_policy_rules()
        for record in records:
            rules = extractor.extract_policy_rules(
                {"coverage_rules": record.coverage_rules}
            )
            extend_policy_rules(rules)
    if records:
        print(f"[startup] Loaded {len(records)} policy rule set(s) into cache.")


def _model_loaded() -> bool:
    from modules.services.ml_pipeline import router as ml_router
    return ml_router is not None


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title    = "PA Validation System API",
        version  = "2.0.0",
        docs_url = "/api/docs",
        redoc_url= "/api/redoc",
        lifespan = lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins_list,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    app.include_router(policies_router)
    app.include_router(submissions_router)
    app.include_router(health_router)

    return app


app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.API_PORT))

    print("PA Validation System API")
    print(f"Docs   -> http://localhost:{port}/api/docs")
    print(f"Health -> http://localhost:{port}/api/health")

    uvicorn.run(
        "app:app",
        host      = settings.API_HOST,
        port      = port,
        reload    = False,        # always off in production
        log_level = "warning",
    )