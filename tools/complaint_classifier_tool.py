"""
complaint_classifier_tool.py
CrewAI Tool yang membungkus model hasil fine-tuning (IndoBERT + LoRA) untuk
mengklasifikasikan kategori tiket komplain secara otomatis. Dipakai CS Agent
SEBELUM memanggil sop_retriever, supaya SOP yang diambil lebih terarah
(bisa dipakai untuk filter metadata source_file berdasarkan kategori).

Ini adalah komponen fine-tuning pada arsitektur (lihat Bab III.3.3.3 laporan),
berbeda dari RAG (sop_retriever) dan model prediktif ML klasik
(machine_failure_predictor) yang sudah ada sebelumnya.
"""

import sys
import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ComplaintClassifierInput(BaseModel):
    complaint_text: str = Field(..., description="Isi teks komplain pelanggan yang ingin diklasifikasikan")


class ComplaintClassifierTool(BaseTool):
    name: str = "complaint_category_classifier"
    description: str = (
        "Mengklasifikasikan teks komplain pelanggan ke salah satu dari 4 kategori "
        "(keterlambatan_pengiriman, produk_cacat, retur_barang, eskalasi_kritikal) "
        "menggunakan model IndoBERT yang telah di-fine-tune dengan LoRA pada histori "
        "tiket komplain PT Cipta Manufaktur Nusantara. Gunakan tool ini di awal untuk "
        "menentukan kategori komplain sebelum mencari SOP yang relevan."
    )
    args_schema: type[BaseModel] = ComplaintClassifierInput

    def _run(self, complaint_text: str) -> str:
        try:
            from finetune.inference import classify_complaint
        except ImportError as e:
            return f"Modul inference tidak dapat dimuat: {e}"

        try:
            result = classify_complaint(complaint_text)
        except FileNotFoundError as e:
            return str(e)
        except Exception as e:
            return (
                f"Gagal memuat/menjalankan model fine-tuned ({type(e).__name__}: {str(e)[:150]}). "
                "Pastikan base model IndoBERT dapat diunduh (butuh akses internet ke HuggingFace Hub "
                "pada saat pertama kali dijalankan) dan LoRA adapter sudah dilatih lewat "
                "finetune/train_lora.py."
            )

        dist_str = ", ".join(f"{k}={v:.1%}" for k, v in result["distribution"].items())
        return (
            f"Kategori prediksi: {result['predicted_category']} "
            f"(confidence: {result['confidence']:.1%})\n"
            f"Distribusi probabilitas semua kelas: {dist_str}"
        )
