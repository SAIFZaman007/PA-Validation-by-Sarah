"""
Simplified ML Model Trainer - No matplotlib dependencies
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

print("=" * 80)
print("ML MODEL TRAINER - SIMPLE VERSION")
print("=" * 80)
print()

# Load data
print("Step 1: Loading data...")
data_path = Path("data/processed/pa_features.csv")
df = pd.read_csv(data_path)

print(f"✓ Loaded {len(df)} samples")
print(f"  Features: {len(df.columns) - 1}")
print(f"  Approvals: {sum(df['label'])} ({sum(df['label'])/len(df)*100:.1f}%)")
print(f"  Denials: {len(df)-sum(df['label'])} ({(len(df)-sum(df['label']))/len(df)*100:.1f}%)")

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']
feature_names = X.columns.tolist()

# Split data
print("\nStep 2: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Train Logistic Regression
print("\n" + "=" * 80)
print("Step 3: Training Logistic Regression")
print("=" * 80)

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

lr_model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"✓ Training complete")
print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Evaluate
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

lr_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr, zero_division=0),
    'recall': recall_score(y_test, y_pred_lr, zero_division=0),
    'f1': f1_score(y_test, y_pred_lr, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
}

cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\nLogistic Regression Results:")
print(f"  Accuracy:  {lr_metrics['accuracy']:.3f}")
print(f"  Precision: {lr_metrics['precision']:.3f}")
print(f"  Recall:    {lr_metrics['recall']:.3f}")
print(f"  F1 Score:  {lr_metrics['f1']:.3f}")
print(f"  ROC AUC:   {lr_metrics['roc_auc']:.3f}")

print(f"\n  Confusion Matrix:")
print(f"    True Neg:  {cm_lr[0][0]:3d}  |  False Pos: {cm_lr[0][1]:3d}")
print(f"    False Neg: {cm_lr[1][0]:3d}  |  True Pos:  {cm_lr[1][1]:3d}")

# Train Decision Tree
print("\n" + "=" * 80)
print("Step 4: Training Decision Tree")
print("=" * 80)

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced'
)

dt_model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"✓ Training complete")
print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Evaluate
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

dt_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt, zero_division=0),
    'recall': recall_score(y_test, y_pred_dt, zero_division=0),
    'f1': f1_score(y_test, y_pred_dt, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_dt)
}

cm_dt = confusion_matrix(y_test, y_pred_dt)

print("\nDecision Tree Results:")
print(f"  Accuracy:  {dt_metrics['accuracy']:.3f}")
print(f"  Precision: {dt_metrics['precision']:.3f}")
print(f"  Recall:    {dt_metrics['recall']:.3f}")
print(f"  F1 Score:  {dt_metrics['f1']:.3f}")
print(f"  ROC AUC:   {dt_metrics['roc_auc']:.3f}")

print(f"\n  Confusion Matrix:")
print(f"    True Neg:  {cm_dt[0][0]:3d}  |  False Pos: {cm_dt[0][1]:3d}")
print(f"    False Neg: {cm_dt[1][0]:3d}  |  True Pos:  {cm_dt[1][1]:3d}")

# Compare models
print("\n" + "=" * 80)
print("Step 5: Model Comparison")
print("=" * 80)

comparison = pd.DataFrame([
    {
        'Model': 'Logistic Regression',
        'Accuracy': f"{lr_metrics['accuracy']:.3f}",
        'Precision': f"{lr_metrics['precision']:.3f}",
        'Recall': f"{lr_metrics['recall']:.3f}",
        'F1': f"{lr_metrics['f1']:.3f}",
        'ROC AUC': f"{lr_metrics['roc_auc']:.3f}"
    },
    {
        'Model': 'Decision Tree',
        'Accuracy': f"{dt_metrics['accuracy']:.3f}",
        'Precision': f"{dt_metrics['precision']:.3f}",
        'Recall': f"{dt_metrics['recall']:.3f}",
        'F1': f"{dt_metrics['f1']:.3f}",
        'ROC AUC': f"{dt_metrics['roc_auc']:.3f}"
    }
])

print(comparison.to_string(index=False))

# Select best model
if lr_metrics['f1'] >= dt_metrics['f1']:
    best_model = lr_model
    best_name = 'logistic_regression'
    best_metrics = lr_metrics
    print(f"\n✓ Best model: Logistic Regression (F1: {lr_metrics['f1']:.3f})")
else:
    best_model = dt_model
    best_name = 'decision_tree'
    best_metrics = dt_metrics
    print(f"\n✓ Best model: Decision Tree (F1: {dt_metrics['f1']:.3f})")

# Feature importance
print("\n" + "=" * 80)
print("Step 6: Feature Importance")
print("=" * 80)

if best_name == 'decision_tree':
    importances = dt_model.feature_importances_
else:
    importances = np.abs(lr_model.coef_[0])

feature_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_imp.head(10).to_string(index=False))

# Save model
print("\n" + "=" * 80)
print("Step 7: Saving Model")
print("=" * 80)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / f"{best_name}_model.joblib"

model_data = {
    'model': best_model,
    'feature_names': feature_names,
    'metrics': best_metrics,
    'model_name': best_name
}

joblib.dump(model_data, model_path)
print(f"✓ Saved model to: {model_path}")

# Save feature importance
feature_imp_path = models_dir / "feature_importance.csv"
feature_imp.to_csv(feature_imp_path, index=False)
print(f"✓ Saved feature importance to: {feature_imp_path}")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\nBest Model: {best_name}")
print(f"Accuracy: {best_metrics['accuracy']:.1%}")
print(f"F1 Score: {best_metrics['f1']:.3f}")
print(f"\nModel saved to: {model_path}")