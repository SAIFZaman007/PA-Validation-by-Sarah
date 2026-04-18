"""
Central configuration for the PA Validation System backend.
"""

from pathlib import Path
from typing  import List

from pydantic          import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).parent.parent.absolute()

class Settings(BaseSettings):
    """
    Application settings, read from environment variables or a .env file.
    Every field has a sensible development default.
    """

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",   # extra .env keys are silently ignored
    )

    # ── Security ──────────────────────────────────────────────────────────────
    PA_SECRET_KEY: str = "dev-only-change-in-production"

    # ── Database ──────────────────────────────────────────────────────────────
    # Default: SQLite inside backend/ — zero config for local development.
    # Override with a PostgreSQL URL in .env for production.
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BACKEND_DIR / 'pa_validation.db'}"

    # PostgreSQL connection pool — ignored when using SQLite.
    DB_POOL_SIZE:    int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    # ── API server ────────────────────────────────────────────────────────────
    API_HOST:  str  = "0.0.0.0"
    API_PORT:  int  = 8000
    API_DEBUG: bool = True

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
    @property
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

# ── NLP settings ────────────────
NLP_CONFIG = {
    "max_features": 1000,
    "min_df":       2,
    "max_df":       0.8,
    "ngram_range":  (1, 2),
}

# ── ML training settings ────────────────────────────────────
ML_CONFIG = {
    "test_size":    0.2,
    "random_state": 42,
    "cv_folds":     5,
}

# ── Routing confidence thresholds ───────────────────────────
ROUTER_THRESHOLDS = {
    "high_confidence": 0.80,
    "low_confidence":  0.50,
}

# ── Medical code regex ──────────────────────────────────────
CPT_PATTERN   = r'\b\d{5}\b'
ICD10_PATTERN = r'\b[A-Z]\d{2}(?:\.\d{1,4})?\b'

# ── ML model artefacts ────────────────────────────────────────────────────────
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Logs ──────────────────────────────────────────────────────────────────────
LOGS_DIR = BACKEND_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Module-level singleton 
settings = Settings()