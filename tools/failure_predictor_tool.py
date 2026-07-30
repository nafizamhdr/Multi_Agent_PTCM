"""
failure_predictor_tool.py
CrewAI Tool untuk prediksi kegagalan mesin dari data sensor terbaru,
menggunakan model yang sudah dilatih (train_failure_model.py). Dipakai Maintenance Agent.
"""

import os
import joblib
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class FailurePredictionInput(BaseModel):
    machine_id: str = Field(..., description="ID mesin, misal CNC-01")
    air_temperature_K: float = Field(..., description="Suhu udara sekitar dalam Kelvin")
    process_temperature_K: float = Field(..., description="Suhu proses dalam Kelvin")
    rotational_speed_rpm: float = Field(..., description="Kecepatan rotasi dalam RPM")
    torque_Nm: float = Field(..., description="Torsi dalam Newton-meter")
    tool_wear_min: float = Field(..., description="Durasi pemakaian tool dalam menit")


class FailurePredictorTool(BaseTool):
    name: str = "machine_failure_predictor"
    description: str = (
        "Memprediksi apakah sebuah mesin berisiko mengalami kegagalan (dan jenis kegagalan apa) "
        "berdasarkan data sensor terkini (suhu, kecepatan rotasi, torsi, tool wear). "
        "Model dilatih dari histori sensor mesin PT Cipta Manufaktur Nusantara."
    )
    args_schema: type[BaseModel] = FailurePredictionInput

    def _run(self, machine_id: str, air_temperature_K: float, process_temperature_K: float,
              rotational_speed_rpm: float, torque_Nm: float, tool_wear_min: float) -> str:
        model_path = os.path.join(MODEL_DIR, "failure_model.joblib")
        if not os.path.exists(model_path):
            return "Model belum dilatih. Jalankan train_failure_model.py terlebih dahulu."

        model = joblib.load(model_path)
        le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

        X = pd.DataFrame([{
            "air_temperature_K": air_temperature_K,
            "process_temperature_K": process_temperature_K,
            "rotational_speed_rpm": rotational_speed_rpm,
            "torque_Nm": torque_Nm,
            "tool_wear_min": tool_wear_min,
        }])[feature_names]

        pred_encoded = model.predict(X)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]

        proba = model.predict_proba(X)[0]
        top_idx = proba.argsort()[::-1][:3]
        proba_str = ", ".join(f"{le.classes_[i]}={proba[i]:.2%}" for i in top_idx)

        return (
            f"Prediksi untuk mesin {machine_id}: {pred_label}\n"
            f"Distribusi probabilitas (top-3): {proba_str}"
        )
