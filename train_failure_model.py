"""
train_failure_model.py (versi dengan penanganan imbalance)
Training model prediktif kegagalan mesin dari data AI4I2020 ASLI, dengan
perbandingan: (1) baseline tanpa penanganan imbalance, (2) dengan SMOTE
(Synthetic Minority Oversampling Technique) pada data latih.

PENTING: SMOTE hanya diterapkan pada data LATIH (bukan data uji), supaya
evaluasi tetap mencerminkan distribusi kelas asli di dunia nyata -- praktik
standar untuk menghindari data leakage/evaluasi yang bias optimistis.

Usage:
    python train_failure_model.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "maintenance", "sensor_log.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURES = [
    "air_temperature_K", "process_temperature_K",
    "rotational_speed_rpm", "torque_Nm", "tool_wear_min",
]


def train_and_eval(X_train, y_train, X_test, y_test, label, le):
    results = {}

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["random_forest"] = {
        "model": rf,
        "accuracy": accuracy_score(y_test, rf_pred),
        "f1_macro": f1_score(y_test, rf_pred, average="macro"),
    }

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric="mlogloss"
    )
    xgb_clf.fit(X_train, y_train)
    xgb_pred = xgb_clf.predict(X_test)
    results["xgboost"] = {
        "model": xgb_clf,
        "accuracy": accuracy_score(y_test, xgb_pred),
        "f1_macro": f1_score(y_test, xgb_pred, average="macro"),
    }

    print(f"\n=== [{label}] Perbandingan Model ===")
    for name, r in results.items():
        print(f"{name}: accuracy={r['accuracy']:.3f}, f1_macro={r['f1_macro']:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = results[best_name]["model"]
    best_pred = best_model.predict(X_test)
    print(f"[BEST] {best_name} (f1_macro={results[best_name]['f1_macro']:.3f})")
    print(classification_report(y_test, best_pred, target_names=le.classes_, zero_division=0))

    return best_name, best_model, results[best_name]["f1_macro"]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]

    le = LabelEncoder()
    y = le.fit_transform(df["failure_type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 1. Baseline: tanpa penanganan imbalance ---
    print("=" * 60)
    print("BASELINE (data latih asli, tanpa penanganan imbalance)")
    print("=" * 60)
    base_name, base_model, base_f1 = train_and_eval(X_train, y_train, X_test, y_test, "Baseline", le)

    # --- 2. Dengan SMOTE pada data latih ---
    print("\n" + "=" * 60)
    print("DENGAN SMOTE (oversampling kelas minoritas pada data latih)")
    print("=" * 60)
    min_class_count = np.bincount(y_train).min()
    k_neighbors = max(1, min(5, min_class_count - 1))
    print(f"(k_neighbors SMOTE disesuaikan ke {k_neighbors} karena kelas minoritas terkecil "
          f"hanya punya {min_class_count} sampel di data latih)")

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"Jumlah data latih sebelum SMOTE: {len(X_train)}, sesudah: {len(X_train_sm)}")

    smote_name, smote_model, smote_f1 = train_and_eval(X_train_sm, y_train_sm, X_test, y_test, "SMOTE", le)

    if smote_f1 > base_f1:
        final_name, final_model, final_f1 = smote_name, smote_model, smote_f1
        approach = "SMOTE"
    else:
        final_name, final_model, final_f1 = base_name, base_model, base_f1
        approach = "Baseline"

    print(f"\n{'=' * 60}")
    print(f"MODEL FINAL: {final_name} ({approach}), f1_macro={final_f1:.3f}")
    print(f"{'=' * 60}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, os.path.join(MODEL_DIR, "failure_model.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_names.joblib"))
    print(f"\n[SAVED] Model tersimpan di {MODEL_DIR}/ (final={final_name}, approach={approach})")


if __name__ == "__main__":
    main()
