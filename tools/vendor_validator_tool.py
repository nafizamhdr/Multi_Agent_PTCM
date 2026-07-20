"""
vendor_validator_tool.py
CrewAI Tool untuk cek status kelengkapan dokumen vendor terhadap checklist standar.
Dipakai Vendor/HR Agent.
"""

import os
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vendor_hr")


class VendorLookupInput(BaseModel):
    vendor_id_or_name: str = Field(
        ..., description="ID vendor (misal VND-003) atau nama vendor (bisa parsial, misal 'Baja Mandiri')"
    )


class VendorValidatorTool(BaseTool):
    name: str = "vendor_document_validator"
    description: str = (
        "Mengecek status kelengkapan dokumen sebuah vendor (ISO, NPWP, SIUP, kontrak, NDA, dll) "
        "berdasarkan checklist standar perusahaan, dan memberi tahu dokumen apa saja yang masih kurang."
    )
    args_schema: type[BaseModel] = VendorLookupInput

    def _run(self, vendor_id_or_name: str) -> str:
        vendors = pd.read_csv(os.path.join(DATA_DIR, "vendor_profiles.csv"))
        checklist = pd.read_csv(os.path.join(DATA_DIR, "document_checklist_standar.csv"))

        match = vendors[
            vendors["vendor_id"].str.contains(vendor_id_or_name, case=False, na=False)
            | vendors["nama_vendor"].str.contains(vendor_id_or_name, case=False, na=False)
        ]

        if match.empty:
            return f"Vendor '{vendor_id_or_name}' tidak ditemukan di database."

        row = match.iloc[0]
        missing = row["dokumen_belum_lengkap"]

        checklist_notes = []
        if missing != "-":
            for doc in missing.split("; "):
                doc = doc.strip()
                ref = checklist[checklist["dokumen"] == doc]
                if not ref.empty:
                    checklist_notes.append(f"  - {doc}: {ref.iloc[0]['catatan']}")

        result = (
            f"Vendor: {row['nama_vendor']} ({row['vendor_id']})\n"
            f"Kategori: {row['kategori']}\n"
            f"Kelengkapan dokumen: {row['kelengkapan_dokumen_pct']}%\n"
            f"Status: {row['status_validasi']}\n"
            f"Skor risiko: {row['skor_risiko']}\n"
            f"Dokumen belum lengkap: {missing}\n"
        )
        if checklist_notes:
            result += "Catatan checklist:\n" + "\n".join(checklist_notes)

        return result
