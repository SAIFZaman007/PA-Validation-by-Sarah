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

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

from core.config   import settings
from core.database import create_tables, migrate_tables

from modules.routes.policies      import router as policies_router
from modules.routes.prescriptions import router as prescriptions_router
from modules.routes.submissions   import router as submissions_router
from modules.routes.health        import router as health_router

# Keep SQLAlchemy and watchfiles quiet
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Step 1: add any missing columns to existing tables (idempotent)
    await migrate_tables()
    # Step 2: create any tables that don't exist yet
    await create_tables()
    # Step 3: warm policy rules cache
    await _warm_policy_cache()
    print("[startup] Database ready.")
    print(f"[startup] ML Model: {'loaded' if _model_loaded() else 'NOT loaded — run: python src/ml/train_model.py'}")
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
        title     = "PA Validation System API",
        version   = "3.0.1",
        docs_url  = "/api/docs",
        redoc_url = "/api/redoc",
        lifespan  = lifespan,
        description=(
            "Prior Authorisation Validation System — ML-powered insurance PA routing.\n\n"
            "**Flow:**\n"
            "1. `POST /api/generate-policy` or `POST /api/upload-policy` — create a policy.\n"
            "2. `POST /api/generate-prescription` or `POST /api/upload-prescription` — create a prescription.\n"
            "3. `POST /api/submit-request` — upload prescription PDF + select insurer → get ML validation report.\n"
            "4. `GET /api/download/request/{result_id}` — download the PDF report.\n"
            "5. `GET /api/stats` — live routing statistics."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins_list,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    app.include_router(policies_router)
    app.include_router(prescriptions_router)
    app.include_router(submissions_router)
    app.include_router(health_router)

    return app


app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.API_PORT))

    print("=" * 60)
    print("PA Validation System API  v3.0.1")
    print("=" * 60)
    print(f"Docs   -> http://localhost:{port}/api/docs")
    print(f"Health -> http://localhost:{port}/api/health")
    print(f"Stats  -> http://localhost:{port}/api/stats")

    try:
        uvicorn.run(
            "app:app",
            host      = settings.API_HOST,
            port      = port,
            reload    = False,
            log_level = "warning",
        )
    except KeyboardInterrupt:
        print("Server stopped manually.")