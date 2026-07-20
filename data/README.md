# Dataset Dummy — Studi Kasus PT Cipta Manufaktur Nusantara

Dataset ini dibuat untuk mendukung development dan evaluasi sistem multi-agent final project.
Semua data adalah **data sintetis (dummy)**, dibuat dengan seed acak agar reproducible.

## Struktur Folder

```
dataset_dummy/
├── maintenance/
│   ├── sensor_log.csv              # data sensor mesin (400 baris) — untuk model prediktif (RF/XGBoost)
│   └── maintenance_history_notes.csv  # catatan naratif teknisi (60 kasus) — untuk embedding + vector DB
│
├── customer_service/
│   ├── sop/                        # 4 dokumen SOP (markdown) — corpus RAG
│   │   ├── SOP_keterlambatan_pengiriman.md
│   │   ├── SOP_produk_cacat.md
│   │   ├── SOP_retur_barang.md
│   │   └── SOP_eskalasi_komplain.md
│   └── complaint_tickets.csv       # 120 tiket komplain historis — untuk semantic search & evaluasi
│
├── vendor_hr/
│   ├── vendor_profiles.csv         # 20 profil vendor dengan status kelengkapan dokumen
│   └── document_checklist_standar.csv  # checklist standar validasi dokumen (referensi agent)
│
└── finance/
    └── weekly_aggregated_report.csv  # agregasi mingguan biaya, DIHITUNG dari data maintenance + CS
                                       # (bukti bahwa Finance Agent bergantung pada output agent lain)
```

## Cara Pakai per Teknik LLM

| Teknik | File yang dipakai | Catatan |
|---|---|---|
| **Fine-tuning / model prediktif** | `maintenance/sensor_log.csv` | Fitur mirip AI4I 2020 (air/process temperature, rotational speed, torque, tool wear) → target `machine_failure` (binary) dan `failure_type` (multi-class) |
| **RAG** | `customer_service/sop/*.md` | Chunking per section (##), embed, simpan ke vector DB. Query CS masuk → retrieve SOP relevan → generate jawaban |
| **Embedding + Vector DB** | `maintenance/maintenance_history_notes.csv`, `customer_service/complaint_tickets.csv` | Embed kolom teks (`catatan_teknisi`, `isi_komplain`) untuk cari kasus historis serupa saat query baru masuk |
| **Agregasi lintas-agent** | `finance/weekly_aggregated_report.csv` | Contoh output yang seharusnya dihasilkan Finance Agent secara dinamis dari data 3 divisi lain, bukan file statis — file ini hanya starting point/ground truth untuk testing |

## Catatan Penting

- Semua nominal biaya dalam Rupiah (IDR), dibuat dengan asumsi wajar untuk skala UMKM-menengah manufaktur.
- Distribusi kelas pada `sensor_log.csv` sengaja dibuat imbalanced (mayoritas "No Failure") untuk mensimulasikan kondisi nyata seperti pada dataset AI4I 2020.
- Dataset ini BUKAN untuk klaim akurasi model final — gunakan sebagai starting point, silakan diperbesar volumenya (misal jadi 2000-5000 baris) kalau butuh training model yang lebih robust.
