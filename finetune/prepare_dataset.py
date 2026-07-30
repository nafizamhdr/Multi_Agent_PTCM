"""
prepare_dataset.py
Menyiapkan dataset untuk fine-tuning classifier kategori komplain dari
data/customer_service/complaint_tickets.csv.

Label yang dipelajari: kategori (keterlambatan_pengiriman, produk_cacat,
retur_barang, eskalasi_kritikal) -- 4 kelas, diambil dari kolom `kategori`
yang sudah berlabel asli (bukan label buatan/pseudo-label).

Output: finetune/data/train.csv dan finetune/data/val.csv (stratified split
80/20), plus finetune/data/label_map.json untuk mapping label <-> index.

Usage:
    python finetune/prepare_dataset.py
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "data", "customer_service", "complaint_tickets.csv")
OUT_DIR = os.path.join(BASE_DIR, "finetune", "data")


def main():
    df = pd.read_csv(SRC_PATH)
    df = df[["isi_komplain", "kategori"]].rename(columns={"isi_komplain": "text", "kategori": "label"})

    labels = sorted(df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    print("Distribusi label:")
    print(df["label"].value_counts())
    print(f"\nTotal sampel: {len(df)}, jumlah kelas: {len(labels)}")

    # stratified split supaya proporsi tiap kategori terjaga di train & val
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] train.csv: {len(train_df)} baris, val.csv: {len(val_df)} baris")
    print(f"[OK] label_map.json: {label2id}")


if __name__ == "__main__":
    main()
