## Cara Menjalankan

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

# 3b. Fine-tuning classifier kategori komplain (LoRA + IndoBERT) -- butuh akses
#     internet untuk download base model IndoBERT saat pertama kali dijalankan
python finetune/prepare_dataset.py
python finetune/train_lora.py

# 3c. (Opsional) Upgrade embedding dari TF-IDF ke sentence-transformers -- edit
#     .env: EMBEDDING_BACKEND=sentence-transformers, lalu index ulang:
python ingest_data.py

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
├── llm_config.py               # konfigurasi Ollama
├── ingest_data.py             # index SOP & case history ke ChromaDB
├── map_ai4i2020.py            # mapping dataset AI4I2020 asli -> skema sensor_log.csv
├── train_failure_model.py     # training baseline vs SMOTE (RF & XGBoost)
├── requirements.txt
├── .env.example
│
├── tools/                     # 7 CrewAI tool
│   ├── vector_store.py        # wrapper ChromaDB, dual-backend embedding (TF-IDF / sentence-transformers)
│   ├── embedding_smoke_test.py    # verifikasi pipeline sentence-transformers tanpa download
│   ├── rag_tool.py            # SOPRetrieverTool (CS Agent)
│   ├── case_search_tool.py    # CaseSearchTool (Maintenance & CS Agent)
│   ├── complaint_classifier_tool.py  # ComplaintClassifierTool, model hasil fine-tuning (CS Agent)
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
├── finetune/                  # pipeline fine-tuning LoRA (IndoBERT)
│   ├── prepare_dataset.py     # split train/val dari complaint_tickets.csv
│   ├── train_lora.py          # training LoRA (--smoke-test untuk verifikasi offline)
│   ├── inference.py           # load adapter untuk klasifikasi komplain baru
│   └── lora_adapter/          # output adapter hasil training (dibuat otomatis)
│
├── evaluator/
│   ├── evaluate.py            # 5 metrik: accuracy, efficiency, explainability, hallucination-proxy, effectiveness
│   └── test_queries.json      # contoh query (termasuk yang butuh >1 agent)
│
├── data/                      # dataset (maintenance: AI4I2020 asli; CS/vendor/finance: dummy)
├── models/                    # model .joblib hasil training
└── chroma_db/                 # vector DB (dibuat otomatis oleh ingest_data.py)
```

## Catatan Penting untuk Laporan

1. **Dataset maintenance sekarang data ASLI** (AI4I 2020, UCI), bukan sintetis lagi.
   Distribusi kegagalan sangat imbalanced (~3,4% failure rate) -- realistis untuk data
   dunia nyata. `train_failure_model.py` membandingkan baseline vs SMOTE secara eksplisit;
   SMOTE menaikkan F1-macro (0,592 -> 0,611) dengan trade-off presisi turun pada kelas
   minoritas, dinamika yang wajar dibahas di laporan.

2. **LLM backend**: Ollama lokal (`llama3.1`). Sempat dicoba Groq dan Gemini (API cloud
   gratis) tapi keduanya kena rate limit saat dipakai testing berulang -- Ollama dipilih
   final karena jalan lokal tanpa batasan rate limit, meski perlu resource laptop sendiri
   dan sedikit lebih lambat/kurang stabil untuk function-calling dibanding model besar
   cloud (lihat temuan bug di laporan Bab V).

3. **Fine-tuning classifier komplain**: LoRA pada IndoBERT, sudah diverifikasi lewat
   `python finetune/train_lora.py --smoke-test`. Untuk model asli, jalankan tanpa flag
   tersebut (butuh akses internet ke HuggingFace Hub saat pertama kali).

4. **Hierarchical process CrewAI**: manager (orchestrator) di-generate otomatis oleh
   framework berdasarkan `manager_llm`. Kalau butuh kontrol routing yang lebih eksplisit,
   pertimbangkan custom manager agent untuk versi lanjutan.

5. **Bukti interaksi antar-agent**: lihat `evaluator/test_queries.json` — ada query yang
   butuh >1 agent (misal Maintenance → Finance). Ini yang perlu didemokan ke dosen sebagai
   bukti sistem bukan sekadar beberapa chatbot terpisah.
