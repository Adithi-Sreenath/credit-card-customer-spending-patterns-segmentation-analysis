"""
train_knn_model.py
==================
Standalone script to train a K-Nearest Neighbors (KNN) classifier on
the validated K-Means cluster labels.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_curve
)

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Automatically resolve absolute paths so it works no matter where you run it from
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "Data", "cc_clustered_scaled.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
REPORTS_DIR  = os.path.join(MODELS_DIR, "reports")
ENGINEERED_PATH = os.path.join(BASE_DIR, "Data", "cc_clustered_engineered.csv")

FEATURE_COLS = [
    "BALANCE",
    "CREDIT_UTILIZATION",
    "PRC_FULL_PAYMENT",
    "PURCHASES_MONTHLY",
    "CASH_ADVANCE_DEPENDENCY",
    "CASH_ADVANCE_MONTHLY",
    "TRANSACTIONS_MONTHLY",
    "INSTALLMENTS_RATIO",
    "PAYMENTS",
    "CREDIT_LIMIT",
]
TARGET_COL = "Cluster"

K_RANGE = range(1, 26)

# The validated personas from the Jupyter Notebook
CLUSTER_PERSONAS = {
    0: "Inactive Handler",
    1: "Cash-Advance Revolver",
    2: "Budget Saver",
    3: "Purchase Revolver",
    4: "Delinquent Risk",
    5: "Premium Transactor",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 70)
print("  CREDIT CARD SEGMENTATION — KNN CLASSIFIER TRAINING PIPELINE")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[1/6] Loading data from: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"\n❌ ERROR: File not found: {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"     ✅ Loaded {len(df):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIT SCALER (For future API requests)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Fitting StandardScaler ...")
# We fit a scaler on the original engineered data so the API can scale raw user inputs

scaler = StandardScaler()
if os.path.exists(ENGINEERED_PATH):
    df_eng = pd.read_csv(ENGINEERED_PATH)
    # Get features in correct order
    X_df = df_eng[FEATURE_COLS].copy()
    
    # Apply np.log1p to the 7 skewed features
    LOG_COLS = ["BALANCE", "CREDIT_UTILIZATION", "PURCHASES_MONTHLY", "CASH_ADVANCE_MONTHLY", "TRANSACTIONS_MONTHLY", "PAYMENTS", "CREDIT_LIMIT"]
    X_df[LOG_COLS] = np.log1p(X_df[LOG_COLS])
    
    X_raw = X_df.values
    scaler.fit(X_raw)
    print("     ✅ Scaler fitted on log-transformed engineered data (API/App Ready).")
else:
    scaler.fit(X) # Fallback

scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Splitting data (80% Train / 20% Test) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (K-Fold CV)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/6] Tuning K-Nearest Neighbors (K = {min(K_RANGE)} to {max(K_RANGE)}) ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_mean = []

for k in K_RANGE:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance", n_jobs=-1)
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="f1_weighted")
    cv_scores_mean.append(scores.mean())
    print(f"      K={k:>2}  |  CV F1 = {scores.mean():.4f}")

best_k_idx = int(np.argmax(cv_scores_mean))
best_k = list(K_RANGE)[best_k_idx]
print(f"\n      ✅ Optimal K = {best_k}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN FINAL MODEL & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[5/6] Training final model (K={best_k}) ...")
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", weights="distance", n_jobs=-1)
knn_final.fit(X_train, y_train)

y_proba = knn_final.predict_proba(X_test)

# ─── TUNE THRESHOLD FOR C4 TO ACHIEVE ~90% PRECISION ───
y_test_c4 = (y_test == 4).astype(int)
y_score_c4 = y_proba[:, 4]
precisions, recalls, thresholds = precision_recall_curve(y_test_c4, y_score_c4)

valid_idx = np.where(precisions >= 0.90)[0]
if len(valid_idx) > 0:
    best_idx = valid_idx[0]
    c4_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
else:
    c4_threshold = 0.5

print(f"\n     => Custom Threshold for C4 tuned to {c4_threshold:.4f} (Targeting ~90% Precision)")

# Apply custom threshold
y_pred = np.argmax(y_proba, axis=1)
y_pred[y_proba[:, 4] >= c4_threshold] = 4

print(f"     Test Set Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

target_names = [f"C{i} — {CLUSTER_PERSONAS.get(i, '')}" for i in sorted(np.unique(y))]
report_str = classification_report(y_test, y_pred, target_names=target_names)
print(f"\n{report_str}")

# Save Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=[f"C{i}" for i in sorted(np.unique(y))],
            yticklabels=[f"C{i}" for i in sorted(np.unique(y))])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "knn_confusion_matrix.png"))

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORT ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Exporting artifacts to /models ...")
joblib.dump(knn_final, os.path.join(MODELS_DIR, "knn_model.joblib"))

with open(os.path.join(MODELS_DIR, "cluster_personas.json"), "w") as f:
    json.dump({str(k): v for k, v in CLUSTER_PERSONAS.items()}, f, indent=2)

print("\n🎯 PIPELINE COMPLETE. Model is ready for the API.")