"""
map_ai4i2020.py
Memetakan dataset AI4I 2020 (UCI, asli/nyata) ke skema sensor_log.csv yang
dipakai sistem. Berbeda dari versi sintetis sebelumnya, dataset ini TIDAK
direkayasa -- label kegagalan apa adanya dari data asli, termasuk tingkat
imbalance yang realistis (~3,4% failure rate).

Mapping machine_id: AI4I2020 punya kolom 'Type' (L/M/H, varian kualitas
produk), dipetakan ke 6 machine_id narasi kita (CNC=H, CONV=M, PRESS=L,
konsisten dengan desain osf_threshold sebelumnya) secara round-robin supaya
data tersebar merata ke "6 mesin" sesuai studi kasus, sambil Product ID asli
tetap disimpan untuk ketertelusuran.

failure_type diturunkan dari kolom flag TWF/HDF/PWF/OSF/RNF (prioritas
berurutan sesuai dokumentasi resmi AI4I2020, karena beberapa baris punya
lebih dari satu flag aktif bersamaan).

Usage:
    python map_ai4i2020.py
"""

import pandas as pd
import numpy as np

SRC_PATH = "/mnt/user-data/uploads/ai4i2020.csv"
OUT_PATH = "/home/claude/multi_agent_ptcm/data/maintenance/sensor_log.csv"

type_to_machines = {
    "H": ["CNC-01", "CNC-02", "CNC-03"],
    "M": ["CONV-01", "CONV-02"],
    "L": ["PRESS-01"],
}

FAILURE_PRIORITY = [
    ("TWF", "Tool Wear Failure"),
    ("HDF", "Heat Dissipation Failure"),
    ("PWF", "Power Failure"),
    ("OSF", "Overstrain Failure"),
    ("RNF", "Random Failure"),
]


def derive_failure_type(row):
    if row["Machine failure"] == 0:
        return "No Failure"
    for flag_col, label in FAILURE_PRIORITY:
        if row[flag_col] == 1:
            return label
    # Edge case pada AI4I2020 asli: machine_failure=1 tapi semua flag=0
    # (didokumentasikan terjadi pada sebagian kecil baris di dataset asli)
    return "Random Failure"


def assign_machine_id(row, counters):
    candidates = type_to_machines[row["Type"]]
    idx = counters[row["Type"]] % len(candidates)
    counters[row["Type"]] += 1
    return candidates[idx]


def main():
    df = pd.read_csv(SRC_PATH)
    df.columns = [c.strip() for c in df.columns]

    counters = {"H": 0, "M": 0, "L": 0}
    df["machine_id"] = df.apply(lambda r: assign_machine_id(r, counters), axis=1)
    df["failure_type"] = df.apply(derive_failure_type, axis=1)

    out = pd.DataFrame({
        "record_id": [f"REC-{i+1:05d}" for i in range(len(df))],
        "machine_id": df["machine_id"],
        "original_product_id": df["Product ID"],
        "timestamp": pd.date_range("2026-01-01", periods=len(df), freq="h"),
        "air_temperature_K": df["Air temperature [K]"],
        "process_temperature_K": df["Process temperature [K]"],
        "rotational_speed_rpm": df["Rotational speed [rpm]"],
        "torque_Nm": df["Torque [Nm]"],
        "tool_wear_min": df["Tool wear [min]"],
        "machine_failure": df["Machine failure"],
        "failure_type": df["failure_type"],
    })

    print("=== Distribusi kelas (data AI4I2020 ASLI, tidak direkayasa) ===")
    print(out["failure_type"].value_counts())
    print(f"\nTotal baris: {len(out)}")
    print(f"Persentase failure: {out['machine_failure'].mean():.2%}")
    print(f"\nDistribusi machine_id:")
    print(out["machine_id"].value_counts())

    out.to_csv(OUT_PATH, index=False)
    print(f"\n[SAVED] {OUT_PATH}")


if __name__ == "__main__":
    main()
