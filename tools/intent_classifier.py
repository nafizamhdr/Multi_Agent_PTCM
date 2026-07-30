"""
intent_classifier.py
Classifier intent berbasis keyword (deterministik, TANPA LLM) untuk menentukan
agent mana yang relevan terhadap sebuah query, sebelum delegasi dilakukan.

Kenapa keyword-based, bukan LLM-based?
Pada pengujian sebelumnya (lihat Bab V.4 laporan), delegasi yang sepenuhnya
mengandalkan penalaran bebas LLM (lewat tool delegate_work_to_coworker bawaan
CrewAI) terbukti menghasilkan beberapa masalah: bug argumen tool yang salah
ketik ('coworkk'), delegasi konkuren yang gagal, dan yang paling parah --
untuk query yang di luar cakupan sistem (misal "berapa gaji karyawan"),
orchestrator tetap memaksa mendelegasikan ke semua agent dan berujung pada
jawaban yang di-hallucinate sepenuhnya.

Dengan classifier berbasis keyword yang dijalankan SEBELUM agent manapun
dipanggil, query yang tidak cocok dengan domain manapun bisa langsung ditolak
dengan pesan baku tanpa melibatkan LLM sama sekali -- menghilangkan risiko
hallucination pada kasus di luar cakupan secara total (bukan cuma menguranginya).
"""

import re

AGENT_KEYWORDS = {
    "maintenance": [
        "mesin", "cnc", "conveyor", "press", "rusak", "kerusakan", "error",
        "kegagalan", "gagal", "tool wear", "downtime", "perbaikan", "maintenance",
        "overheat", "suhu", "rotasi", "torsi", "sensor", "breakdown",
    ],
    "cs": [
        "komplain", "keluhan", "pelanggan", "customer", "sop", "retur",
        "keterlambatan", "pengiriman", "produk cacat", "cacat", "refund",
        "kompensasi", "tiket", "cs", "customer service",
    ],
    "vendor": [
        "vendor", "supplier", "pemasok", "dokumen", "iso", "kontrak", "nda",
        "procurement", "onboarding", "sertifikat", "checklist",
    ],
    "finance": [
        "biaya", "anggaran", "keuangan", "finance", "operasional", "kerugian",
        "rugi", "untung", "estimasi biaya", "laporan keuangan", "pengeluaran",
    ],
}

# Pola ID eksplisit yang bisa dipakai untuk exact-match lookup (lihat tools/case_search_tool.py)
ID_PATTERNS = {
    "ticket": re.compile(r"\bTCK-\d{4,6}\b", re.IGNORECASE),
    "maintenance_case": re.compile(r"\bMTC-\d{3,6}\b", re.IGNORECASE),
    "po_number": re.compile(r"\bPO-\d{4,6}\b", re.IGNORECASE),
    "sensor_record": re.compile(r"\bREC-\d{4,6}\b", re.IGNORECASE),
    "vendor_id": re.compile(r"\bVND-\d{2,4}\b", re.IGNORECASE),
}


def classify_intent(query: str) -> list[str]:
    """
    Mengembalikan daftar agent key ('maintenance', 'cs', 'vendor', 'finance')
    yang relevan dengan query, berdasarkan kemunculan keyword. Daftar kosong
    berarti query tidak cocok dengan domain manapun -- HARUS ditolak tanpa
    memanggil agent/LLM apapun (lihat crew.py run_query()).

    Memakai word-boundary matching (\\b), bukan substring biasa, supaya
    keyword pendek seperti 'nda' (untuk mendeteksi 'NDA') tidak salah cocok
    dengan potongan kata lain seperti 'ditunda' -- bug nyata yang ditemukan
    saat pengujian awal classifier ini.
    """
    query_lower = query.lower()
    matched = []
    for agent_key, keywords in AGENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                matched.append(agent_key)
                break
    return matched


def extract_ids(query: str) -> dict:
    """Mengekstrak ID eksplisit (nomor tiket, PO, dsb) dari query untuk exact-match lookup."""
    found = {}
    for id_type, pattern in ID_PATTERNS.items():
        matches = pattern.findall(query)
        if matches:
            found[id_type] = [m.upper() for m in matches]
    return found


def order_agents(matched: list[str]) -> list[str]:
    """
    Menentukan urutan eksekusi agent. Finance selalu dieksekusi PALING AKHIR
    kalau muncul bersama agent lain, karena Finance Agent secara struktural
    membutuhkan output Maintenance/CS sebagai konteks (lihat Bab II.2.2 & III.3.2
    laporan mengenai ketergantungan fungsional Finance Agent).
    """
    priority = {"maintenance": 0, "cs": 0, "vendor": 0, "finance": 1}
    return sorted(matched, key=lambda a: priority.get(a, 0))
