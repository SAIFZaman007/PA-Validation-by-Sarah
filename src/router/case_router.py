"""
Confidence-based Case Router for the PA Validation System.

Loads a trained sklearn model from models/ and routes each PA case to:
  - auto_approve   (high confidence approval)
  - auto_deny      (high confidence denial)
  - manual_review  (low or medium confidence)
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

# Make backend root importable so core.config can be found.
# src/router/case_router.py  ->  src/  ->  backend/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import MODELS_DIR, ROUTER_THRESHOLDS


class CaseRouter:
    """Route PA cases based on prediction confidence from the ML model."""

    def __init__(self, model_path: Path = None):
        """
        Load the trained model from disk.

        Args:
            model_path: Explicit path to a .joblib file.
                        When omitted, searches MODELS_DIR for the logistic
                        regression model first, then any .joblib file.
        """
        if model_path is None:
            preferred = MODELS_DIR / "logistic_regression_model.joblib"
            if preferred.exists():
                model_path = preferred
            else:
                candidates = sorted(MODELS_DIR.glob("*.joblib"))
                if not candidates:
                    raise FileNotFoundError(
                        f"No trained model found in {MODELS_DIR}. "
                        "Run: python src/ml/train_model_simple.py"
                    )
                model_path = candidates[0]

        print(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)

        self.model         = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.model_name    = model_data["model_name"]
        self.high_threshold= ROUTER_THRESHOLDS["high_confidence"]
        self.low_threshold = ROUTER_THRESHOLDS["low_confidence"]

        # Three-line startup summary — shows exactly what was loaded and at
        # what thresholds, which is useful for verifying config in the console.
        print(f"  Loaded {self.model_name} model")
        print(f"  High confidence threshold: {self.high_threshold}")
        print(f"  Low confidence threshold:  {self.low_threshold}")

    # ── Core prediction ───────────────────────────────────────────────────────

    def predict_with_confidence(self, features_dict: dict) -> dict:
        """
        Run the model and return prediction + probability scores.

        Returns:
            {
                prediction:          0 or 1,
                prediction_label:    "Approve" | "Deny",
                confidence:          float,
                approve_probability: float,
                deny_probability:    float,
            }
        """
        features_df   = pd.DataFrame([features_dict])[self.feature_names]
        prediction    = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]
        confidence    = probabilities[prediction]

        return {
            "prediction":          int(prediction),
            "prediction_label":    "Approve" if prediction == 1 else "Deny",
            "confidence":          float(confidence),
            "approve_probability": float(probabilities[1]),
            "deny_probability":    float(probabilities[0]),
        }

    # ── Routing decision ──────────────────────────────────────────────────────

    def route_case(self, features_dict: dict) -> dict:
        """
        Assign a routing tier based on prediction confidence.

        Returns:
            {
                tier, action, color, confidence, explanation,
                prediction, approve_probability, deny_probability
            }
        """
        pred       = self.predict_with_confidence(features_dict)
        confidence = pred["confidence"]
        label      = pred["prediction_label"]   # "Approve" or "Deny"

        if confidence >= self.high_threshold:
            if label == "Approve":
                tier, action, color = "auto_approve", "AUTO-APPROVE", "green"
                explanation = f"High confidence ({confidence:.1%}) approval prediction"
            else:
                tier, action, color = "auto_deny", "AUTO-DENY", "red"
                explanation = f"High confidence ({confidence:.1%}) denial prediction"
        elif confidence >= self.low_threshold:
            tier, action, color = "manual_review", "MANUAL REVIEW", "yellow"
            explanation = f"Medium confidence ({confidence:.1%}) - needs human review"
        else:
            tier, action, color = "manual_review", "MANUAL REVIEW", "orange"
            explanation = f"Low confidence ({confidence:.1%}) - needs careful review"

        return {
            "tier":                tier,
            "action":              action,
            "color":               color,
            "confidence":          confidence,
            "explanation":         explanation,
            "prediction":          label,
            "approve_probability": pred["approve_probability"],
            "deny_probability":    pred["deny_probability"],
        }

    # ── Batch statistics ──────────────────────────────────────────────────────

    def get_routing_statistics(self, features_list: list) -> dict:
        """Return aggregate routing stats for a list of feature dicts."""
        routings      = [self.route_case(f) for f in features_list]
        total         = len(routings)
        auto_approve  = sum(1 for r in routings if r["tier"] == "auto_approve")
        auto_deny     = sum(1 for r in routings if r["tier"] == "auto_deny")
        manual_review = sum(1 for r in routings if r["tier"] == "manual_review")

        return {
            "total_cases":       total,
            "auto_approve":      auto_approve,
            "auto_approve_pct":  auto_approve  / total * 100 if total else 0,
            "auto_deny":         auto_deny,
            "auto_deny_pct":     auto_deny     / total * 100 if total else 0,
            "manual_review":     manual_review,
            "manual_review_pct": manual_review / total * 100 if total else 0,
            "automation_rate":   (auto_approve + auto_deny) / total * 100 if total else 0,
        }