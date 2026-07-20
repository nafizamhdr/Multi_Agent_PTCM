# Multi-Agent System — PT Cipta Manufaktur Nusantara

Skeleton sistem multi-agent untuk final project (CrewAI + Ollama + ChromaDB).

## Status Testing

Semua komponen di bawah ini **sudah diuji dan berjalan** tanpa perlu Ollama (dites di
lingkungan sandbox tanpa akses internet ke Ollama registry):

| Komponen | Status | Cara test |
|---|---|---|
| Ingestion data ke ChromaDB (RAG + case search) | ✅ Berjalan | `python ingest_data.py` |
| Training model prediktif (RF vs XGBoost) | ✅ Berjalan | `python train_failure_model.py` |
| Semua tool (RAG, case search, predictor, vendor validator, finance summary) | ✅ Berjalan | lihat `evaluator/evaluate.py` |
| Crew (4 agent + orchestrator) berhasil dirakit | ✅ Berjalan | `python -c "from crew import build_crew; build_crew('test')"` |
| Evaluator tool-level (accuracy, efficiency, explainability, hallucination-proxy) | ✅ Berjalan, hasil nyata | `python evaluator/evaluate.py` |
| **Eksekusi penuh crew.kickoff() dengan reasoning LLM** | ⏳ **Belum dites** — butuh Ollama jalan | `python main.py "query kamu"` |

**Kenapa belum dites end-to-end penuh?** Sandbox coding saya tidak punya akses internet ke
Ollama/model registry (whitelist domain saya cuma pypi/npm/github), jadi saya tidak bisa
menjalankan `ollama serve` di sini. Semua bagian yang TIDAK butuh LLM (vector DB, model
prediktif, tool logic, evaluator) sudah saya jalankan dan verifikasi hasilnya nyata (lihat
`evaluator/evaluate.py` output). Bagian yang butuh Ollama (orchestrator delegation,
sintesis jawaban akhir) sudah saya tulis lengkap dan crew berhasil dirakit tanpa error,
tapi baru bisa dijalankan penuh (`crew.kickoff()`) di laptop kamu yang sudah ada Ollama.

## Cara Menjalankan di Laptop Kamu

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Setup Ollama (kalau belum ada)
#    Download dari https://ollama.com
ollama pull llama3.1
ollama serve                     # jalankan di terminal terpisah

# 3. Siapkan data & model (sekali saja, atau setiap dataset berubah)
python ingest_data.py
python train_failure_model.py

# 4. Jalankan evaluator tool-level (tidak butuh Ollama, opsional tapi bagus buat cek)
python evaluator/evaluate.py

# 5. Jalankan sistem
python main.py "Mesin CNC-01 sering error karena tool wear, berapa estimasi biaya kalau ditunda perbaikannya bulan depan?"
```

## Struktur Project

```
multi_agent_ptcm/
├── main.py                    # entry point
├── crew.py                    # orchestrator + hierarchical delegation
├── llm_config.py              # konfigurasi Ollama
├── ingest_data.py             # index SOP & case history ke ChromaDB
├── train_failure_model.py     # training RF vs XGBoost
├── requirements.txt
├── .env.example
│
├── tools/                     # 5 CrewAI tool
│   ├── vector_store.py        # wrapper ChromaDB + TF-IDF embedding
│   ├── rag_tool.py            # SOPRetrieverTool (CS Agent)
│   ├── case_search_tool.py    # CaseSearchTool (Maintenance & CS Agent)
│   ├── failure_predictor_tool.py  # FailurePredictorTool (Maintenance Agent)
│   ├── vendor_validator_tool.py   # VendorValidatorTool (Vendor Agent)
│   └── finance_summary_tool.py    # FinanceSummaryTool (Finance Agent)
│
├── agents/                    # 4 definisi Agent CrewAI
│   ├── maintenance_agent.py
│   ├── cs_agent.py
│   ├── vendor_agent.py
│   └── finance_agent.py
│
├── evaluator/
│   ├── evaluate.py            # 4 metrik: accuracy, efficiency, explainability, hallucination-proxy
│   └── test_queries.json      # contoh query (termasuk yang butuh >1 agent)
│
├── data/                      # dataset dummy (copy dari sesi sebelumnya)
├── models/                    # model .joblib hasil training
└── chroma_db/                 # vector DB (dibuat otomatis oleh ingest_data.py)
```

## Catatan Penting untuk Laporan

1. **Embedding TF-IDF**: dipakai karena keterbatasan resource sandbox (sentence-transformers
   butuh ~2GB untuk torch). Untuk laporan/demo final, pertimbangkan upgrade ke
   `sentence-transformers` (kode swap sudah didokumentasikan di `tools/vector_store.py`) —
   kualitas retrieval semantik akan jauh lebih baik daripada TF-IDF yang berbasis kata kunci.

2. **Model prediktif Maintenance Agent** akurasinya masih rendah (f1_macro ~0.23) karena
   dataset dummy kecil (400 baris) dan imbalanced. Ini realistis — sama seperti temuan di
   project AURA. Untuk laporan, bahas ini sebagai limitation & rencana improvement
   (misal SMOTE untuk handle imbalance, atau perbesar dataset).

3. **Hierarchical process CrewAI**: manager (orchestrator) di-generate otomatis oleh
   framework berdasarkan `manager_llm`. Kalau butuh kontrol routing yang lebih eksplisit
   (dan lebih mudah dijelaskan ke dosen), pertimbangkan custom manager agent atau migrasi
   ke LangGraph untuk versi lanjutan.

4. **Bukti interaksi antar-agent**: lihat `evaluator/test_queries.json` — ada 3 dari 5
   contoh query yang butuh >1 agent (misal Maintenance → Finance). Ini yang perlu
   didemokan ke dosen sebagai bukti sistem bukan sekadar 4 chatbot terpisah.
