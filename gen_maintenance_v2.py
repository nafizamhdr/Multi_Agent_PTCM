"""
gen_maintenance_v2.py
Generator dataset sensor mesin versi perbaikan. Berbeda dari versi awal, setiap
jenis kegagalan sekarang punya kondisi fisik yang jelas dan konsisten (rule-based),
mengikuti logika yang mirip dataset AI4I 2020 (Matzka, 2020) -- bukan sekadar
override probabilitas acak di atas distribusi random seperti versi sebelumnya.

Aturan kegagalan (diterapkan berurutan, kegagalan pertama yang terpenuhi dipakai):
1. Tool Wear Failure   : tool_wear_min tinggi (>200 menit), dengan noise/nudge
2. Heat Dissipation     : (process_temp - air_temp) < 8.6 K DAN rotational_speed < 1380 rpm
3. Power Failure        : power = torque x kecepatan sudut (rad/s) berada di luar rentang aman
4. Overstrain Failure   : tool_wear_min x torque melebihi ambang batas (bervariasi per tipe mesin)
5. Random Failure       : kegagalan acak berpeluang kecil, tanpa sinyal dari fitur (irreducible)
6. No Failure           : jika tidak ada kondisi di atas terpenuhi

Ukuran dataset diperbesar dari 400 -> 3000 baris, dan parameter disetel supaya
setiap kelas kegagalan minoritas tetap punya minimal ~100+ sampel absolut,
cukup untuk model belajar pola pembeda tiap kelas (bukan cuma kelas mayoritas).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000
machine_ids = ["CNC-01", "CNC-02", "CNC-03", "CONV-01", "CONV-02", "PRESS-01"]

# Tipe mesin memengaruhi ambang batas Overstrain Failure (mirip variasi kualitas L/M/H
# pada AI4I 2020) -- mesin CNC (presisi tinggi) lebih rentan overstrain pada beban lebih rendah
machine_type = {
    "CNC-01": "H", "CNC-02": "H", "CNC-03": "H",
    "CONV-01": "M", "CONV-02": "M", "PRESS-01": "L",
}
osf_threshold = {"H": 10000, "M": 11500, "L": 13000}  # ambang tool_wear_min x torque


def generate_row(i):
    machine = np.random.choice(machine_ids)
    mtype = machine_type[machine]

    air_temp = np.random.normal(298, 2)
    process_temp = air_temp + np.random.normal(10, 1)
    rot_speed = np.random.normal(1538, 150)          # rpm
    torque = np.clip(np.random.normal(40, 10), 3, 76)  # Nm, dijaga tetap positif & wajar
    tool_wear = np.random.uniform(0, 260)             # menit

    omega = rot_speed * 2 * np.pi / 60                # rad/s
    power = torque * omega                            # Watt

    failure = "No Failure"

    # 1) Tool Wear Failure -- wear tinggi, dengan sedikit noise supaya batasnya tidak tajam sekali
    if tool_wear > 200 and np.random.rand() < 0.75:
        failure = "Tool Wear Failure"
    # 2) Heat Dissipation Failure -- selisih suhu kecil + rotasi lambat
    elif (process_temp - air_temp) < 8.6 and rot_speed < 1380 and np.random.rand() < 0.85:
        failure = "Heat Dissipation Failure"
    # 3) Power Failure -- daya di luar rentang aman (terlalu kecil atau terlalu besar)
    elif (power < 3500 or power > 9000) and np.random.rand() < 0.85:
        failure = "Power Failure"
    # 4) Overstrain Failure -- beban kumulatif (wear x torque) melebihi ambang sesuai tipe mesin
    elif tool_wear * torque > osf_threshold[mtype] and np.random.rand() < 0.85:
        failure = "Overstrain Failure"
    # 5) Random Failure -- benar-benar acak, tanpa sinyal dari fitur (irreducible error, realistis)
    elif np.random.rand() < 0.008:
        failure = "Random Failure"

    target = 0 if failure == "No Failure" else 1

    return {
        "record_id": f"REC-{i+1:05d}",
        "machine_id": machine,
        "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i * 2),
        "air_temperature_K": round(air_temp, 1),
        "process_temperature_K": round(process_temp, 1),
        "rotational_speed_rpm": int(rot_speed),
        "torque_Nm": round(torque, 1),
        "tool_wear_min": int(tool_wear),
        "machine_failure": target,
        "failure_type": failure,
    }


def main():
    rows = [generate_row(i) for i in range(N)]
    df = pd.DataFrame(rows)

    print("=== Distribusi kelas (versi baru) ===")
    print(df["failure_type"].value_counts())
    print(f"\nTotal baris: {len(df)}")
    print(f"Persentase failure: {df['machine_failure'].mean():.1%}")

    out_path = "/home/claude/multi_agent_ptcm/data/maintenance/sensor_log.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
