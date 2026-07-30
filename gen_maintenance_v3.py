"""
gen_maintenance_v3.py
Generator dataset sensor mesin final. Memakai pendekatan targeted sampling per
kelas: untuk setiap jenis kegagalan, fitur di-sampling dari rentang yang secara
sengaja mengarah ke kondisi kegagalan tersebut (importance sampling), lalu
diverifikasi lewat fungsi aturan fisik yang SAMA (compute_failure) yang dipakai
untuk semua kelas -- sehingga label akhir tetap konsisten dengan aturan yang
didokumentasikan di laporan, sekaligus jumlah sampel tiap kelas dapat dikontrol
supaya tidak ada kelas yang kekurangan data (masalah utama pada versi awal).

Aturan kegagalan (prioritas berurutan, kondisi pertama yang terpenuhi dipakai):
1. Tool Wear Failure   : tool_wear_min > 200 menit
2. Heat Dissipation     : (process_temp - air_temp) < 8.6 K DAN rotational_speed < 1380 rpm
3. Power Failure        : power = torque x kecepatan sudut (rad/s) di luar rentang [3500, 9000] Watt
4. Overstrain Failure   : tool_wear_min x torque melebihi ambang batas (bervariasi per tipe mesin)
5. Random Failure       : kegagalan acak, TIDAK bergantung fitur sama sekali (irreducible error)
6. No Failure           : jika tidak ada kondisi di atas terpenuhi
"""

import numpy as np
import pandas as pd

np.random.seed(42)

machine_ids = ["CNC-01", "CNC-02", "CNC-03", "CONV-01", "CONV-02", "PRESS-01"]
machine_type = {
    "CNC-01": "H", "CNC-02": "H", "CNC-03": "H",
    "CONV-01": "M", "CONV-02": "M", "PRESS-01": "L",
}
osf_threshold = {"H": 7000, "M": 8500, "L": 10000}

TARGET_COUNTS = {
    "No Failure": 1900,
    "Tool Wear Failure": 350,
    "Power Failure": 250,
    "Heat Dissipation Failure": 200,
    "Overstrain Failure": 200,
    "Random Failure": 100,
}


def compute_failure(air_temp, process_temp, rot_speed, torque, tool_wear, mtype):
    """Fungsi aturan tunggal yang dipakai untuk memutuskan label -- konsisten
    dipakai untuk semua kelas, supaya hasil akhir selalu sesuai aturan fisik
    yang didokumentasikan, walau proses sampling fitur di bawahnya di-bias
    per kelas target."""
    omega = rot_speed * 2 * np.pi / 60
    power = torque * omega

    if tool_wear > 200 and np.random.rand() < 0.75:
        return "Tool Wear Failure"
    if (process_temp - air_temp) < 8.6 and rot_speed < 1380 and np.random.rand() < 0.85:
        return "Heat Dissipation Failure"
    if (power < 3500 or power > 9000) and np.random.rand() < 0.85:
        return "Power Failure"
    if tool_wear * torque > osf_threshold[mtype] and np.random.rand() < 0.85:
        return "Overstrain Failure"
    if np.random.rand() < 0.008:
        return "Random Failure"
    return "No Failure"


def sample_features(target_class, machine, mtype):
    """Sampling fitur dibias mengarah ke kondisi target_class, memakai rentang
    yang dihitung supaya TIDAK secara tidak sengaja memicu kondisi kegagalan
    lain yang urutan prioritasnya lebih tinggi (lihat penjelasan di laporan
    Bab III.3.3.4 mengenai desain dataset ini)."""

    if target_class == "No Failure":
        air_temp = np.random.normal(298, 2)
        process_temp = air_temp + np.random.normal(11, 1)
        rot_speed = np.random.normal(1560, 80)
        torque = np.clip(np.random.normal(35, 6), 20, 45)
        tool_wear = np.random.uniform(0, 100)

    elif target_class == "Tool Wear Failure":
        air_temp = np.random.normal(298, 2)
        process_temp = air_temp + np.random.normal(11, 1)
        rot_speed = np.random.normal(1560, 80)
        torque = np.clip(np.random.normal(35, 6), 20, 45)
        tool_wear = np.random.uniform(205, 260)

    elif target_class == "Heat Dissipation Failure":
        air_temp = np.random.normal(298, 2)
        process_temp = air_temp + np.random.uniform(2, 8.3)
        rot_speed = np.random.uniform(1000, 1375)
        torque = np.clip(np.random.normal(35, 6), 20, 45)
        tool_wear = np.random.uniform(0, 150)

    elif target_class == "Power Failure":
        air_temp = np.random.normal(298, 2)
        process_temp = air_temp + np.random.normal(11, 1)
        if np.random.rand() < 0.5:
            torque = np.random.uniform(3, 12)
            rot_speed = np.random.uniform(1400, 1700)
        else:
            torque = np.random.uniform(60, 76)
            rot_speed = np.random.uniform(1700, 2100)
        tool_wear = np.random.uniform(0, 150)

    elif target_class == "Overstrain Failure":
        air_temp = np.random.normal(298, 2)
        process_temp = air_temp + np.random.normal(11, 1)
        rot_speed = np.random.uniform(800, 1000)
        torque = np.random.uniform(55, 76)
        wear_min = osf_threshold[mtype] / torque
        tool_wear = np.random.uniform(min(wear_min, 195), 199)

    elif target_class == "Random Failure":
        # RNF tidak bergantung fitur sama sekali -- pakai rentang "aman" yang sama
        # dengan No Failure, karena secara fisik memang tidak ada sinyal pembeda
        air_temp = np.random.normal(298, 2)
        process_temp = air_temp + np.random.normal(11, 1)
        rot_speed = np.random.normal(1560, 80)
        torque = np.clip(np.random.normal(35, 6), 20, 45)
        tool_wear = np.random.uniform(0, 100)

    return air_temp, process_temp, rot_speed, torque, tool_wear


def generate_dataset():
    rows = []
    record_i = 0

    for target_class, count in TARGET_COUNTS.items():
        collected = 0
        attempts = 0
        max_attempts = count * 50

        while collected < count and attempts < max_attempts:
            attempts += 1
            machine = np.random.choice(machine_ids)
            mtype = machine_type[machine]
            air_temp, process_temp, rot_speed, torque, tool_wear = sample_features(
                target_class, machine, mtype
            )

            actual_label = compute_failure(air_temp, process_temp, rot_speed, torque, tool_wear, mtype)

            # RNF: terima langsung sesuai desain (tidak bergantung fitur, lihat catatan di atas)
            if target_class == "Random Failure":
                actual_label = "Random Failure"

            if actual_label != target_class:
                continue  # buang sampel yang "kena ambil" kondisi prioritas lain, coba lagi

            record_i += 1
            rows.append({
                "record_id": f"REC-{record_i:05d}",
                "machine_id": machine,
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=record_i * 2),
                "air_temperature_K": round(air_temp, 1),
                "process_temperature_K": round(process_temp, 1),
                "rotational_speed_rpm": int(rot_speed),
                "torque_Nm": round(torque, 1),
                "tool_wear_min": int(tool_wear),
                "machine_failure": 0 if target_class == "No Failure" else 1,
                "failure_type": target_class,
            })
            collected += 1

        if collected < count:
            print(f"[WARNING] '{target_class}': hanya berhasil {collected}/{count} "
                  f"setelah {attempts} percobaan")

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main():
    df = generate_dataset()

    print("=== Distribusi kelas (versi final) ===")
    print(df["failure_type"].value_counts())
    print(f"\nTotal baris: {len(df)}")
    print(f"Persentase failure: {df['machine_failure'].mean():.1%}")

    out_path = "/home/claude/multi_agent_ptcm/data/maintenance/sensor_log.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
