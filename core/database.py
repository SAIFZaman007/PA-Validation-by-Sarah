"""
Database engine and session management.

Development  -> SQLite via aiosqlite  (zero-config, file on disk)
Production   -> PostgreSQL via asyncpg (set DATABASE_URL in .env)
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.config import settings


# ── Engine ────────────────────────────────────────────────────────────────────
def _build_engine():
    """
    Build the async engine with correct pool settings per database type.

    SQLite: single-connection StaticPool (no concurrent write support).
    PostgreSQL: full configurable pool from .env settings.

    echo is always False here — SQLAlchemy SQL logging is controlled via
    the logging module in app.py, not via the engine flag, so it doesn't
    flood the console during normal development.
    """
    url = settings.DATABASE_URL

    if url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool
        return create_async_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )

    return create_async_engine(
        url,
        pool_size    = settings.DB_POOL_SIZE,
        max_overflow = settings.DB_MAX_OVERFLOW,
        pool_timeout = settings.DB_POOL_TIMEOUT,
        pool_pre_ping= True,   # drops stale connections automatically
        echo         = False,
    )

engine = _build_engine()

# Session factory — used in get_db() and Alembic migrations.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # avoids lazy-load errors after commit
    autocommit=False,
    autoflush=False,
)

# ── Base class for all ORM models ─────────────────────────────────────────────
class Base(DeclarativeBase):
    """
    Shared declarative base. Every model in models.py inherits from this.
    Alembic reads Base.metadata to detect schema changes.
    """
    pass

# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields one AsyncSession per request.

    Usage in a route:
        async def my_route(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(MyModel))

    The session is closed automatically when the request finishes.
    Any unhandled exception triggers a rollback before the close.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# ── Startup helper ────────────────────────────────────────────────────────────
async def create_tables() -> None:
    """
    Create all tables that don't exist yet.

    Called once at application startup via the lifespan context in app.py.
    In production, use `alembic upgrade head` instead for controlled migrations.
    """
    async with engine.begin() as conn:
        from core import models  # noqa: F401 — registers models with Base.metadata
        await conn.run_sync(Base.metadata.create_all)