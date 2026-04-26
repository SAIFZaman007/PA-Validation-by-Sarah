"""
Central configuration for the PA Validation System backend.

Environment variables are read from .env (local dev) or from the platform
environment (Render production). Every field has a sensible default.
"""

from pathlib import Path
from typing  import List

from pydantic          import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/core/config.py
BACKEND_DIR = Path(__file__).parent.parent.absolute()


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    PA_SECRET_KEY: str = "dev-only-change-in-production"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BACKEND_DIR / 'pa_validation.db'}"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def fix_postgres_scheme(cls, v: str) -> str:
        """
        Auto-convert postgresql:// → postgresql+asyncpg://

        Render (and most PaaS providers) supply a plain postgresql:// URL.
        SQLAlchemy's async engine requires the +asyncpg driver suffix.
        This validator patches it transparently so no manual editing is needed.
        """
        if v.startswith("postgresql://") or v.startswith("postgres://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        return v

    # PostgreSQL connection pool — ignored by SQLite
    DB_POOL_SIZE:    int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    # ── API server ────────────────────────────────────────────────────────────
    API_HOST:  str  = "0.0.0.0"
    API_PORT:  int  = 8000
    API_DEBUG: bool = False      # always False in production

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str) -> str:
        return v

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # ── Uploads ───────────────────────────────────────────────────────────────
    @property                          # ← FIX: was a plain method, not a property
    def upload_dir(self) -> Path:
        d = BACKEND_DIR / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def policies_upload_dir(self) -> Path:
        d = self.upload_dir / "policies"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def requests_upload_dir(self) -> Path:
        d = self.upload_dir / "requests"
        d.mkdir(parents=True, exist_ok=True)
        return d

    ALLOWED_EXTENSIONS: set = {"pdf", "txt", "doc", "docx"}
    MAX_FILE_SIZE_MB:   int = 16

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


# ── Constants ──────────────────────────────────────

NLP_CONFIG = {
    "max_features": 1000,
    "min_df":       2,
    "max_df":       0.8,
    "ngram_range":  (1, 2),
}

ML_CONFIG = {
    "test_size":    0.2,
    "random_state": 42,
    "cv_folds":     5,
}

ROUTER_THRESHOLDS = {
    "high_confidence": 0.80,
    "low_confidence":  0.50,
}

CPT_PATTERN   = r'\b\d{5}\b'
ICD10_PATTERN = r'\b[A-Z]\d{2}(?:\.\d{1,4})?\b'

MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

LOGS_DIR = BACKEND_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

settings = Settings()