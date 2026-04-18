"""
ML pipeline service — loads and exposes the four pipeline components.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.nlp.data_generator     import SyntheticDataGenerator
from src.nlp.policy_extractor   import PolicyExtractor
from src.ml.feature_engineering import FeatureEngineer
from src.router.case_router     import CaseRouter

# ── Singleton instances — one per worker process ──────────────────────────────
generator = SyntheticDataGenerator()
extractor = PolicyExtractor()
engineer  = FeatureEngineer()

try:
    router = CaseRouter()
except FileNotFoundError as exc:
    router = None
    print(f"[ML][WARNING] Model not loaded — {exc}")
    print("[ML]          Run: python src/ml/train_model_simple.py")

# ── In-memory policy rules cache ─────────────────────────────────────────────
#
# Extracted policy rules are cached here so the ML pipeline never hits the
# database on every single PA submission.
#
# Lifecycle:
#   - Populated at startup (app.py lifespan) from all Policy rows in the DB.
#   - Extended whenever a new policy is generated or uploaded.
#   - Reset on server restart (use Redis for multi-worker deployments).
#
_policy_rules: list = []

def get_policy_rules() -> list:
    """Return the current in-memory rule cache."""
    return _policy_rules

def extend_policy_rules(new_rules: list) -> None:
    """Append newly extracted rules to the cache."""
    _policy_rules.extend(new_rules)

def clear_policy_rules() -> None:
    """Wipe the cache — used during testing and startup re-warm."""
    _policy_rules.clear()