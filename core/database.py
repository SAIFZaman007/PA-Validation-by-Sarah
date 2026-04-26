"""
Database engine and session management.

Development  -> SQLite via aiosqlite  (zero-config, file on disk)
Production   -> PostgreSQL via asyncpg (set DATABASE_URL in .env)
"""

from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.config import settings


# ── Engine ────────────────────────────────────────────────────────────────────

def _build_engine():
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
        pool_size     = settings.DB_POOL_SIZE,
        max_overflow  = settings.DB_MAX_OVERFLOW,
        pool_timeout  = settings.DB_POOL_TIMEOUT,
        pool_pre_ping = True,
        echo          = False,
    )


engine = _build_engine()

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ── Base class for all ORM models ─────────────────────────────────────────────

class Base(DeclarativeBase):
    """
    Shared declarative base. Every model inherits from this.
    Alembic reads Base.metadata to detect schema changes.
    """
    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Schema helpers ────────────────────────────────────────────────────────────

def _dialect_name() -> str:
    """Return the lowercase dialect name: 'sqlite' or 'postgresql'."""
    return engine.dialect.name.lower()


async def _column_exists(conn, table: str, column: str) -> bool:
    """Check whether a column exists — dialect-aware."""
    if _dialect_name() == "sqlite":
        result = await conn.execute(text(f"PRAGMA table_info({table})"))
        return any(row[1] == column for row in result.fetchall())
    else:
        result = await conn.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ), {"t": table, "c": column})
        return result.fetchone() is not None


async def _table_exists(conn, table: str) -> bool:
    """Check whether a table exists — dialect-aware."""
    if _dialect_name() == "sqlite":
        result = await conn.execute(text(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:t"
        ), {"t": table})
    else:
        result = await conn.execute(text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = :t"
        ), {"t": table})
    return result.fetchone() is not None


async def migrate_tables() -> None:
    """
    Idempotent column-level migrations — called at every startup
    before create_tables().

    Adds any columns that were introduced after the initial schema
    without dropping or altering existing data. Safe to run repeatedly
    on both SQLite (dev) and PostgreSQL (production).

    Covers:
      prescriptions      — columns added in v3.0 that may be missing if the
                           table was created before the Alembic migration ran.
      validation_results — policy_id_used, policy_insurer added in v3.0.
    """
    async with engine.begin() as conn:

        # ── prescriptions: add any missing columns ────────────────────────────
        if await _table_exists(conn, "prescriptions"):
            prescriptions_new_cols = {
                "patient_id":                 "VARCHAR(64)",
                "insurance_id":               "VARCHAR(64)",
                "bmi":                        "FLOAT",
                "conservative_therapy_weeks": "INTEGER",
                "imaging_completed":          "BOOLEAN",
                "prerequisites_met":          "JSON",
                "physician_name":             "VARCHAR(128)",
                "physician_specialty":        "VARCHAR(128)",
                "physician_license":          "VARCHAR(64)",
                "approval_likelihood_label":  "VARCHAR(32)",
                "full_data":                  "JSON",
            }
            for col, col_type in prescriptions_new_cols.items():
                if not await _column_exists(conn, "prescriptions", col):
                    await conn.execute(text(
                        f"ALTER TABLE prescriptions ADD COLUMN {col} {col_type}"
                    ))
                    print(f"[migrate] prescriptions.{col} — added")

        # ── validation_results: add policy tracking columns ───────────────────
        if await _table_exists(conn, "validation_results"):
            vr_new_cols = {
                "policy_id_used": "VARCHAR(64)",
                "policy_insurer": "VARCHAR(128)",
            }
            for col, col_type in vr_new_cols.items():
                if not await _column_exists(conn, "validation_results", col):
                    await conn.execute(text(
                        f"ALTER TABLE validation_results ADD COLUMN {col} {col_type}"
                    ))
                    print(f"[migrate] validation_results.{col} — added")


async def create_tables() -> None:
    """
    Create any tables that don't yet exist (CREATE TABLE IF NOT EXISTS).
    Called AFTER migrate_tables() so existing tables get their new
    columns before any new tables are created.
    """
    async with engine.begin() as conn:
        from core import models  # noqa: F401 — registers all models with Base.metadata
        await conn.run_sync(Base.metadata.create_all)