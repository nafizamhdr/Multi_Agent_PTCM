"""
evaluate.py
Evaluator untuk sistem multi-agent. Dipecah 2 lapis:

1. TOOL-LEVEL METRICS (bisa dijalankan tanpa LLM/Ollama sama sekali) -- accuracy model
   prediktif, efficiency (latency), explainability (kelengkapan sitasi RAG),
   hallucination-proxy (relevansi hasil retrieval).
2. END-TO-END METRICS (butuh Ollama jalan) -- accuracy jawaban akhir vs expected_output
   di test_queries.json, dinilai LLM-as-judge. Lihat evaluate_end_to_end() di bawah,
   panggil manual setelah Ollama tersedia.

Usage:
    python evaluator/evaluate.py
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from tools.vector_store import VectorStore
from tools.rag_tool import SOPRetrieverTool
from tools.case_search_tool import CaseSearchTool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


# ------------------------------------------------------------------
# 1. ACCURACY -- model prediktif Maintenance Agent
# ------------------------------------------------------------------
def eval_accuracy_failure_model():
    df = pd.read_csv(os.path.join(DATA_DIR, "maintenance", "sensor_log.csv"))
    features = ["air_temperature_K", "process_temperature_K",
                "rotational_speed_rpm", "torque_Nm", "tool_wear_min"]

    model = joblib.load(os.path.join(MODEL_DIR, "failure_model.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    X = df[features]
    y = le.transform(df["failure_type"])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pred = model.predict(X_test)
    return {
        "metric": "accuracy",
        "component": "maintenance_agent.failure_predictor",
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 3),
        "n_test_samples": len(y_test),
    }


# ------------------------------------------------------------------
# 2. EFFICIENCY -- latency tool call
# ------------------------------------------------------------------
def eval_efficiency(n_calls: int = 10):
    rag_tool = SOPRetrieverTool()
    case_tool = CaseSearchTool()

    queries = [
        "pelanggan komplain barang terlambat sampai",
        "produk cacat saat diterima pelanggan",
        "mesin CNC overheat suhu tinggi",
        "vendor belum lengkap dokumen sertifikasi",
    ]

    latencies_rag, latencies_case = [], []
    for i in range(n_calls):
        q = queries[i % len(queries)]

        t0 = time.perf_counter()
        rag_tool.run(query=q, n_results=3)
        latencies_rag.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        case_tool.run(query=q, n_results=3)
        latencies_case.append((time.perf_counter() - t0) * 1000)

    return {
        "metric": "efficiency",
        "rag_tool_avg_latency_ms": round(sum(latencies_rag) / len(latencies_rag), 2),
        "case_search_avg_latency_ms": round(sum(latencies_case) / len(latencies_case), 2),
        "n_calls": n_calls,
    }


# ------------------------------------------------------------------
# 3. EXPLAINABILITY -- proxy: apakah tiap hasil RAG menyertakan sitasi sumber
# ------------------------------------------------------------------
def eval_explainability():
    rag_tool = SOPRetrieverTool()
    test_queries = [
        "keterlambatan pengiriman", "produk cacat", "retur barang", "eskalasi komplain kritikal",
    ]

    n_with_source = 0
    for q in test_queries:
        result = rag_tool.run(query=q, n_results=2)
        if "[Sumber:" in result:
            n_with_source += 1

    return {
        "metric": "explainability",
        "component": "cs_agent.sop_retriever",
        "pct_responses_with_citation": round(n_with_source / len(test_queries) * 100, 1),
        "n_test_queries": len(test_queries),
    }


# ------------------------------------------------------------------
# 4. HALLUCINATION-PROXY -- relevansi hasil retrieval (jarak di bawah threshold)
#    Proxy sederhana: kalau retrieval tidak menemukan dokumen relevan (jarak tinggi),
#    risiko agent "mengarang" jawaban (hallucinate) jadi lebih tinggi karena tidak
#    ada grounding yang baik.
# ------------------------------------------------------------------
def eval_hallucination_proxy(distance_threshold: float = 1.6):
    store = VectorStore(persist_dir=os.path.join(BASE_DIR, "chroma_db"))
    test_queries = [
        "keterlambatan pengiriman ke pelanggan",
        "produk cacat dimensi tidak sesuai spesifikasi",
        "proses retur barang salah pesan",
        "mesin conveyor mati mendadak",
        "vendor sertifikat ISO belum lengkap",  # query di luar cakupan SOP CS, harus "gagal" grounding
    ]

    grounded = 0
    for q in test_queries:
        res = store.query("sop_documents", q, n_results=1)
        top_distance = res["distances"][0][0] if res["distances"][0] else 999
        if top_distance < distance_threshold:
            grounded += 1

    return {
        "metric": "hallucination_proxy",
        "component": "cs_agent.sop_retriever",
        "pct_grounded_responses": round(grounded / len(test_queries) * 100, 1),
        "note": (
            "Grounded = jarak retrieval di bawah threshold, artinya ada SOP relevan sebagai "
            "dasar jawaban. Response yang TIDAK grounded berisiko lebih tinggi untuk "
            "di-hallucinate oleh LLM karena tidak ada konteks yang cukup relevan."
        ),
        "n_test_queries": len(test_queries),
    }


def run_all_tool_level_metrics():
    results = [
        eval_accuracy_failure_model(),
        eval_efficiency(),
        eval_explainability(),
        eval_hallucination_proxy(),
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


# ------------------------------------------------------------------
# END-TO-END (butuh Ollama) -- placeholder, dijalankan manual setelah Ollama siap
# ------------------------------------------------------------------
def evaluate_end_to_end():
    """
    Jalankan tiap query di test_queries.json lewat crew.run_query(), lalu nilai
    kesesuaian jawaban dengan expected_output pakai LLM-as-judge (Ollama).
    Skeleton fungsi ini disediakan, implementasi scoring detail menyusul setelah
    Ollama terverifikasi jalan (lihat catatan di README project).
    """
    from crew import run_query

    with open(os.path.join(os.path.dirname(__file__), "test_queries.json")) as f:
        test_cases = json.load(f)

    results = []
    for case in test_cases:
        t0 = time.perf_counter()
        answer = run_query(case["query"])
        latency = time.perf_counter() - t0
        results.append({
            "query": case["query"],
            "expected_agents_involved": case["expected_agents"],
            "answer": answer,
            "latency_sec": round(latency, 2),
        })
    return results


if __name__ == "__main__":
    run_all_tool_level_metrics()
