"""
ingest_data.py
Jalankan sekali di awal (atau setiap dataset berubah) untuk mengisi ChromaDB.

Usage:
    python ingest_data.py
"""

import os
import glob
import pandas as pd
from tools.vector_store import VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def ingest_sop_documents(store: VectorStore):
    sop_files = sorted(glob.glob(os.path.join(DATA_DIR, "customer_service", "sop", "*.md")))
    ids, docs, metas = [], [], []

    for path in sop_files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # chunking sederhana per section (split di heading level 2 "## ")
        sections = content.split("\n## ")
        filename = os.path.basename(path)
        for i, section in enumerate(sections):
            text = section if i == 0 else "## " + section
            if len(text.strip()) < 20:
                continue
            ids.append(f"{filename}::chunk-{i}")
            docs.append(text.strip())
            metas.append({"source_file": filename, "chunk_index": i})

    store.index_documents("sop_documents", ids, docs, metas)
    print(f"[OK] Indexed {len(docs)} chunk SOP dari {len(sop_files)} dokumen")


def ingest_case_history(store: VectorStore):
    maint = pd.read_csv(os.path.join(DATA_DIR, "maintenance", "maintenance_history_notes.csv"))
    tickets = pd.read_csv(os.path.join(DATA_DIR, "customer_service", "complaint_tickets.csv"))

    ids, docs, metas = [], [], []

    for _, row in maint.iterrows():
        ids.append(f"maint::{row['case_id']}")
        docs.append(
            f"[{row['machine_id']}] {row['failure_type']}: {row['catatan_teknisi']}"
        )
        metas.append({
            "type": "maintenance_case",
            "machine_id": row["machine_id"],
            "failure_type": row["failure_type"],
            "downtime_jam": float(row["downtime_jam"]),
            "biaya_perbaikan_idr": float(row["biaya_perbaikan_idr"]),
        })

    for _, row in tickets.iterrows():
        ids.append(f"ticket::{row['ticket_id']}")
        docs.append(f"[{row['kategori']}] {row['isi_komplain']}")
        metas.append({
            "type": "complaint_ticket",
            "kategori": row["kategori"],
            "status_resolusi": row["status_resolusi"],
        })

    store.index_documents("case_history", ids, docs, metas)
    print(f"[OK] Indexed {len(docs)} dokumen case history (maintenance + komplain)")


if __name__ == "__main__":
    store = VectorStore(persist_dir=os.path.join(os.path.dirname(__file__), "chroma_db"))
    ingest_sop_documents(store)
    ingest_case_history(store)
    print("[DONE] Ingestion selesai. ChromaDB siap dipakai oleh agent tools.")
