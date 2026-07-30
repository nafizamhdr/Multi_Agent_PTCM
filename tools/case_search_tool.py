"""
case_search_tool.py
CrewAI Tool untuk mencari kasus historis (maintenance atau komplain CS), memakai
pendekatan HYBRID SEARCH:
    1. Exact-match lookup -- kalau query menyebut ID spesifik (nomor tiket TCK-xxxxx,
       ID kasus maintenance MTC-xxxx, nomor PO, atau record sensor REC-xxxxx), sistem
       langsung mencari baris data yang persis cocok, tanpa melalui similarity search.
       ID semacam ini bersifat string acak tanpa makna semantik, sehingga similarity
       search justru kurang tepat untuk kasus ini (prinsip yang sama dipakai pada
       RAG hybrid di banyak sistem enterprise).
    2. Semantic/TF-IDF search -- dipakai sebagai fallback kalau tidak ada ID eksplisit
       pada query, atau exact-match tidak menemukan hasil.

Dipakai Maintenance Agent & CS Agent.
"""

import os
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tools.vector_store import VectorStore
from tools.intent_classifier import extract_ids

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class CaseSearchInput(BaseModel):
    query: str = Field(..., description="Deskripsi kasus/masalah, atau ID spesifik (nomor tiket/PO/kasus) yang ingin dicari")
    n_results: int = Field(3, description="Jumlah kasus historis yang diambil (dipakai kalau bukan exact-match)")


def _exact_match_lookup(ids: dict) -> str | None:
    """Coba cari baris data yang persis cocok dengan ID yang terdeteksi di query."""
    results = []

    if "ticket" in ids:
        tickets = pd.read_csv(os.path.join(DATA_DIR, "customer_service", "complaint_tickets.csv"))
        for tid in ids["ticket"]:
            row = tickets[tickets["ticket_id"].str.upper() == tid]
            if not row.empty:
                r = row.iloc[0]
                results.append(
                    f"[exact-match, complaint_ticket, ticket_id={tid}]\n"
                    f"[{r['kategori']}] {r['isi_komplain']} "
                    f"(status: {r['status_resolusi']}, waktu_respons: {r['waktu_respons_jam']} jam)"
                )

    if "maintenance_case" in ids:
        cases = pd.read_csv(os.path.join(DATA_DIR, "maintenance", "maintenance_history_notes.csv"))
        for cid in ids["maintenance_case"]:
            row = cases[cases["case_id"].str.upper() == cid]
            if not row.empty:
                r = row.iloc[0]
                results.append(
                    f"[exact-match, maintenance_case, case_id={cid}]\n"
                    f"[{r['machine_id']}] {r['failure_type']}: {r['catatan_teknisi']} "
                    f"(downtime: {r['downtime_jam']} jam, biaya: Rp{r['biaya_perbaikan_idr']:,.0f})"
                )

    if "po_number" in ids:
        tickets = pd.read_csv(os.path.join(DATA_DIR, "customer_service", "complaint_tickets.csv"))
        for po in ids["po_number"]:
            row = tickets[tickets["isi_komplain"].str.upper().str.contains(po, na=False)]
            if not row.empty:
                r = row.iloc[0]
                results.append(
                    f"[exact-match, complaint_ticket via PO number={po}]\n"
                    f"[{r['kategori']}] {r['isi_komplain']} (status: {r['status_resolusi']})"
                )

    if "sensor_record" in ids:
        sensors = pd.read_csv(os.path.join(DATA_DIR, "maintenance", "sensor_log.csv"))
        for rid in ids["sensor_record"]:
            row = sensors[sensors["record_id"].str.upper() == rid]
            if not row.empty:
                r = row.iloc[0]
                results.append(
                    f"[exact-match, sensor_record, record_id={rid}]\n"
                    f"machine_id={r['machine_id']}, failure_type={r['failure_type']}, "
                    f"tool_wear_min={r['tool_wear_min']}, torque_Nm={r['torque_Nm']}"
                )

    return "\n\n---\n\n".join(results) if results else None


class CaseSearchTool(BaseTool):
    name: str = "case_history_search"
    description: str = (
        "Mencari kasus historis (maintenance mesin ATAU komplain pelanggan) yang relevan. "
        "Mendukung pencarian by ID spesifik (nomor tiket TCK-xxxxx, ID kasus MTC-xxxx, "
        "nomor PO, atau record sensor REC-xxxxx) untuk hasil yang presisi, maupun pencarian "
        "berbasis kemiripan makna untuk deskripsi masalah umum."
    )
    args_schema: type[BaseModel] = CaseSearchInput

    def _run(self, query: str, n_results: int = 3) -> str:
        # 1. Coba exact-match dulu kalau ada ID eksplisit di query
        ids = extract_ids(query)
        if ids:
            exact_result = _exact_match_lookup(ids)
            if exact_result:
                return exact_result
            # ID terdeteksi tapi tidak ketemu di data -- tetap lanjut ke semantic search
            # sebagai fallback, jangan langsung menyerah

        # 2. Fallback: semantic/TF-IDF search
        store = VectorStore(persist_dir=CHROMA_DIR)
        results = store.query("case_history", query, n_results=n_results)

        if not results["documents"][0]:
            return "Tidak ditemukan kasus historis yang relevan."

        output = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if k != "type")
            output.append(f"[semantic-search, {meta.get('type')}, skor_jarak={dist:.3f}, {meta_str}]\n{doc}")
        return "\n\n---\n\n".join(output)
