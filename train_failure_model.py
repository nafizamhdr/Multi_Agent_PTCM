"""
train_failure_model.py
Training cepat model klasifikasi kegagalan mesin dari data/maintenance/sensor_log.csv.
Membandingkan Random Forest vs XGBoost (sesuai pendekatan AURA), simpan model terbaik.

Usage:
    python train_failure_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "maintenance", "sensor_log.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURES = [
    "air_temperature_K", "process_temperature_K",
    "rotational_speed_rpm", "torque_Nm", "tool_wear_min",
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]

    le = LabelEncoder()
    y = le.fit_transform(df["failure_type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["random_forest"] = {
        "model": rf,
        "accuracy": accuracy_score(y_test, rf_pred),
        "f1_macro": f1_score(y_test, rf_pred, average="macro"),
    }

    # --- XGBoost ---
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

    print("=== Perbandingan Model ===")
    for name, r in results.items():
        print(f"{name}: accuracy={r['accuracy']:.3f}, f1_macro={r['f1_macro']:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = results[best_name]["model"]
    print(f"\n[BEST MODEL] {best_name} (f1_macro={results[best_name]['f1_macro']:.3f})")

    print("\n=== Classification Report (best model) ===")
    best_pred = best_model.predict(X_test)
    print(classification_report(y_test, best_pred, target_names=le.classes_, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "failure_model.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_names.joblib"))
    print(f"\n[SAVED] Model tersimpan di {MODEL_DIR}/ (best={best_name})")


if __name__ == "__main__":
    main()
