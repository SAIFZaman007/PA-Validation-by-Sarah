"""
ML Model Trainer — PA Validation System.

Primary model:   Logistic Regression (production model, tuned for accuracy + approval rate).
Comparison model: Decision Tree     (used only for performance comparison).

Run from the backend/ directory:
    python src/ml/train_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────

BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

DATA_PATH    = BACKEND_DIR / "data" / "processed" / "pa_features.csv"
MODELS_DIR   = BACKEND_DIR / "models"
SYNTH_JSON   = BACKEND_DIR / "data" / "synthetic" / "synthetic_pa_requests.json"
POLICIES_JSON= BACKEND_DIR / "data" / "synthetic" / "synthetic_policies.json"

MODELS_DIR.mkdir(exist_ok=True)


# ── Generate richer training data if needed ────────────────────────────────────

def _rebuild_training_data() -> pd.DataFrame:
    """
    Build a richer feature matrix from the synthetic JSON files.

    Generates 1,000 samples with a balanced-but-realistic distribution:
      ~45% approve (high/medium likelihood)
      ~55% deny    (low likelihood)
    so the model learns a meaningful decision boundary while being slightly
    more generous than a 50/50 split — yielding a higher real-world approval rate.
    """
    print("[data] Rebuilding training data from synthetic JSON files …")

    from src.nlp.data_generator    import generate_policy, generate_pa_request, INSURERS
    from src.nlp.policy_extractor  import PolicyExtractor
    from src.ml.feature_engineering import FeatureEngineer

    extractor = PolicyExtractor()
    engineer  = FeatureEngineer()

    # Generate 1 policy per insurer to maximise coverage diversity
    policies = [generate_policy(insurer=ins, num_procedures=5) for ins in INSURERS]
    all_rules = []
    for pol in policies:
        rules = extractor.extract_policy_rules(pol)
        all_rules.extend(rules)

    # 1,000 samples: 250 high, 200 medium, 550 low
    # → realistic denial-heavy distribution but still meaningful approval signal
    SAMPLES = [
        ("high",   250),
        ("medium", 200),
        ("low",    550),
    ]

    feature_rows, labels = [], []
    for likelihood, count in SAMPLES:
        for _ in range(count):
            import random
            pol = random.choice(policies)
            pa  = generate_pa_request(policy=pol, approval_likelihood=likelihood)
            feats = engineer.create_features(pa, all_rules)
            feature_rows.append(feats)
            # Ground truth: high/medium → approve (1), low → deny (0)
            labels.append(1 if likelihood in ("high", "medium") else 0)

    df = pd.DataFrame(feature_rows)
    df["label"] = labels

    # Persist for future reruns
    out_path = BACKEND_DIR / "data" / "processed" / "pa_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[data] Saved {len(df)} samples → {out_path}")
    return df


def _load_or_build_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if len(df) >= 500:
            print(f"[data] Loaded {len(df)} existing samples from {DATA_PATH}")
            return df
        print(f"[data] Existing data too small ({len(df)} rows) — rebuilding …")

    return _rebuild_training_data()


# ── Main trainer ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PA Validation System — ML Model Trainer")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = _load_or_build_data()

    X = df.drop("label", axis=1)
    y = df["label"]
    feature_names = X.columns.tolist()

    approve_pct = y.mean() * 100
    deny_pct    = (1 - y.mean()) * 100

    print(f"\n[data] {len(df)} samples  |  Approve: {approve_pct:.1f}%  |  Deny: {deny_pct:.1f}%")
    print(f"[data] Features: {len(feature_names)}")

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[split] Train: {len(X_train)}  |  Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Logistic Regression (PRIMARY model) ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Logistic Regression (Primary Model)")
    print("=" * 70)

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,              # regularisation strength
            max_iter=2000,
            solver="lbfgs",
            class_weight={0: 1.0, 1: 1.3},  # slightly favour approvals
            random_state=42,
            multi_class="auto",
        )),
    ])

    lr_pipeline.fit(X_train, y_train)

    cv_acc = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV Accuracy : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"CV ROC-AUC  : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    y_pred_lr      = lr_pipeline.predict(X_test)
    y_proba_lr     = lr_pipeline.predict_proba(X_test)[:, 1]

    lr_metrics = {
        "accuracy":  accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr, zero_division=0),
        "recall":    recall_score(y_test, y_pred_lr, zero_division=0),
        "f1":        f1_score(y_test, y_pred_lr, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba_lr),
    }
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    print(f"\nTest Results:")
    print(f"  Accuracy : {lr_metrics['accuracy']:.3f}")
    print(f"  Precision: {lr_metrics['precision']:.3f}")
    print(f"  Recall   : {lr_metrics['recall']:.3f}")
    print(f"  F1 Score : {lr_metrics['f1']:.3f}")
    print(f"  ROC AUC  : {lr_metrics['roc_auc']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm_lr[0][0]:4d}  FP: {cm_lr[0][1]:4d}")
    print(f"    FN: {cm_lr[1][0]:4d}  TP: {cm_lr[1][1]:4d}")
    print(f"\n  Approval rate on test set: {y_pred_lr.mean()*100:.1f}%")

    # ── Decision Tree (COMPARISON only) ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Decision Tree (Comparison Model — not used in production)")
    print("=" * 70)

    dt_model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=7,
        class_weight={0: 1.0, 1: 1.2},
        random_state=42,
    )
    dt_model.fit(X_train, y_train)

    cv_acc_dt = cross_val_score(dt_model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV Accuracy : {cv_acc_dt.mean():.3f} ± {cv_acc_dt.std():.3f}")

    y_pred_dt  = dt_model.predict(X_test)
    y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

    dt_metrics = {
        "accuracy":  accuracy_score(y_test, y_pred_dt),
        "precision": precision_score(y_test, y_pred_dt, zero_division=0),
        "recall":    recall_score(y_test, y_pred_dt, zero_division=0),
        "f1":        f1_score(y_test, y_pred_dt, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba_dt),
    }

    print(f"\nTest Results:")
    print(f"  Accuracy : {dt_metrics['accuracy']:.3f}")
    print(f"  Precision: {dt_metrics['precision']:.3f}")
    print(f"  Recall   : {dt_metrics['recall']:.3f}")
    print(f"  F1 Score : {dt_metrics['f1']:.3f}")
    print(f"  ROC AUC  : {dt_metrics['roc_auc']:.3f}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    comparison = pd.DataFrame([
        {"Model": "Logistic Regression (PRIMARY)", **{k: f"{v:.3f}" for k, v in lr_metrics.items()}},
        {"Model": "Decision Tree (comparison)",    **{k: f"{v:.3f}" for k, v in dt_metrics.items()}},
    ])
    print(comparison.to_string(index=False))

    # ── Feature importance (LR uses absolute coefficients) ────────────────────
    lr_coef = lr_pipeline.named_steps["model"].coef_[0]
    feat_imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": np.abs(lr_coef),
    }).sort_values("Importance", ascending=False)

    print("\nTop 10 Feature Importances (Logistic Regression |coef|):")
    print(feat_imp.head(10).to_string(index=False))

    # ── Save models ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving Models")
    print("=" * 70)

    # Primary: Logistic Regression (save pipeline so scaler travels with model)
    lr_path = MODELS_DIR / "logistic_regression_model.joblib"
    joblib.dump({
        "model":        lr_pipeline,
        "feature_names": feature_names,
        "metrics":      lr_metrics,
        "model_name":   "logistic_regression",
    }, lr_path)
    print(f"✓ Saved Logistic Regression → {lr_path}")

    # Comparison: Decision Tree
    dt_path = MODELS_DIR / "decision_tree_model.joblib"
    joblib.dump({
        "model":        dt_model,
        "feature_names": feature_names,
        "metrics":      dt_metrics,
        "model_name":   "decision_tree",
    }, dt_path)
    print(f"✓ Saved Decision Tree       → {dt_path}")

    # Feature importance CSV
    feat_imp_path = MODELS_DIR / "feature_importance.csv"
    feat_imp.to_csv(feat_imp_path, index=False)
    print(f"✓ Saved feature importance  → {feat_imp_path}")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"Primary model : Logistic Regression")
    print(f"  Accuracy    : {lr_metrics['accuracy']:.1%}")
    print(f"  F1 Score    : {lr_metrics['f1']:.3f}")
    print(f"  ROC AUC     : {lr_metrics['roc_auc']:.3f}")
    print(f"  Approval rate (test): {y_pred_lr.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
