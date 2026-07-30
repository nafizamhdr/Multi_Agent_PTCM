"""
main.py
Entry point sistem multi-agent PT Cipta Manufaktur Nusantara.
LLM backend: Ollama (lokal).

Prasyarat sebelum run pertama kali:
    1. python ingest_data.py          -> index SOP & case history ke ChromaDB
    2. python train_failure_model.py  -> latih model prediksi kegagalan mesin
    3. Pastikan Ollama jalan (ollama serve) dan model sudah di-pull:
           ollama pull llama3.1

Usage:
    python main.py "Mesin CNC-01 sering error, apa rekomendasinya?"
"""

import sys
import os

from llm_config import check_llm_available, OLLAMA_MODEL, OLLAMA_BASE_URL


def preflight_check():
    checks = {
        "ChromaDB terisi (jalankan ingest_data.py)": os.path.exists(
            os.path.join(os.path.dirname(__file__), "chroma_db")
        ),
        "Model prediktif tersedia (jalankan train_failure_model.py)": os.path.exists(
            os.path.join(os.path.dirname(__file__), "models", "failure_model.joblib")
        ),
    }
    all_ok = True
    print("=== Preflight check ===")
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'MISSING'}] {name}")
        all_ok = all_ok and ok

    llm_ok = check_llm_available()
    print(f"[{'OK' if llm_ok else 'TIDAK TERJANGKAU'}] Ollama server di {OLLAMA_BASE_URL} (model: {OLLAMA_MODEL})")
    print()
    return all_ok, llm_ok


def main():
    if len(sys.argv) < 2:
        query = "Mesin CNC-01 sering mengalami tool wear, berapa estimasi biaya kalau ditunda perbaikannya bulan depan?"
        print(f"[INFO] Tidak ada argumen query, pakai contoh default:\n  \"{query}\"\n")
    else:
        query = " ".join(sys.argv[1:])

    data_ok, llm_ok = preflight_check()

    if not data_ok:
        print("[STOP] Jalankan 'python ingest_data.py' dan 'python train_failure_model.py' dulu.")
        return

    # Cek out-of-scope duluan -- query di luar cakupan dijawab langsung tanpa
    # butuh LLM sama sekali (lihat tools/intent_classifier.py & crew.py run_query()),
    # jadi tidak perlu menunggu Ollama untuk kasus ini.
    from tools.intent_classifier import classify_intent
    if not classify_intent(query):
        from crew import run_query
        print(f"=== Query ===\n{query}\n")
        print("=== Jawaban Akhir (ditolak sebelum melibatkan agent/LLM manapun) ===")
        print(run_query(query))
        return

    if not llm_ok:
        print(
            "[STOP] Ollama tidak terjangkau di " + OLLAMA_BASE_URL + "\n"
            "Jalankan 'ollama serve' dan 'ollama pull " + OLLAMA_MODEL + "' terlebih dahulu.\n\n"
            "Sementara itu, komponen non-LLM (vector search, model prediktif, evaluator) sudah "
            "bisa diuji lewat:\n"
            "  python evaluator/evaluate.py"
        )
        return

    from crew import run_query
    print(f"=== Query ===\n{query}\n")
    print("=== Menjalankan Orchestrator + Agent ===")
    answer = run_query(query)
    print("\n=== Jawaban Akhir ===")
    print(answer)


if __name__ == "__main__":
    main()
