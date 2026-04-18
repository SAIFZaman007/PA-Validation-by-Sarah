"""
Alembic migration environment — async SQLAlchemy edition.

Reads DATABASE_URL from the same .env that the FastAPI app uses.

Workflow:
  First time:
    alembic revision --autogenerate -m "initial schema"
    alembic upgrade head

  After a model change:
    alembic revision --autogenerate -m "describe the change"
    alembic upgrade head

  Roll back one step:
    alembic downgrade -1
"""

import sys
import asyncio
from pathlib        import Path
from logging.config import fileConfig

from sqlalchemy             import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic                import context

# Make the backend root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config   import settings
from core.database import Base
import core.models  # noqa: F401

alembic_config = context.config
alembic_config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = alembic_config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        alembic_config.get_section(alembic_config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())