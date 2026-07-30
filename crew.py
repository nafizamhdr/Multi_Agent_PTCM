"""
crew.py
Orkestrator kustom untuk sistem multi-agent PT Cipta Manufaktur Nusantara.

PERUBAHAN DARI VERSI SEBELUMNYA (lihat Bab V.4-5.5 laporan untuk latar belakang):
Versi awal memakai Process.hierarchical bawaan CrewAI, di mana manager (LLM)
bebas memutuskan agent mana yang didelegasikan lewat tool delegate_work_to_coworker.
Pendekatan ini terbukti menimbulkan beberapa masalah nyata saat diuji:
    1. Delegasi ke agent yang salah/berlebihan untuk query di luar cakupan
       (contoh: query "berapa gaji karyawan" tetap dipaksa didelegasikan ke
       4 agent, menghasilkan jawaban yang di-hallucinate total)
    2. Bug argumen tool salah ketik oleh LLM ('coworkk' alih-alih 'coworker')
    3. Delegasi konkuren yang gagal ("Executor is already running")
    4. Klaim sitasi sumber yang tidak sesuai tool yang benar-benar dipanggil

Untuk mengatasi ini, orkestrasi diganti total dengan pendekatan yang lebih
eksplisit dan terkontrol (mirip prinsip "delegation tag" pada arsitektur
multi-agent yang lebih matang):
    1. Intent classification berbasis keyword TANPA LLM (tools/intent_classifier.py)
       -- query di luar cakupan ditolak sebelum agent manapun dipanggil, sehingga
       hallucination pada kasus di luar cakupan hilang total, bukan cuma berkurang.
    2. Setiap agent yang relevan dijalankan secara eksplisit lewat Process.sequential
       per-agent (bukan lewat delegate_work_to_coworker), menghilangkan risiko
       argumen tool salah ketik dan delegasi konkuren.
    3. Sitasi sumber pada jawaban akhir dibangun programatik dari tool_call_logger
       (ground truth eksekusi), bukan diminta dari LLM.
"""

from crewai import Crew, Task, Process

from agents.maintenance_agent import build_maintenance_agent
from agents.cs_agent import build_cs_agent
from agents.vendor_agent import build_vendor_agent
from agents.finance_agent import build_finance_agent
from tools.intent_classifier import classify_intent, order_agents, extract_ids
from tool_call_logger import tool_call_logger
from llm_config import get_llm

AGENT_BUILDERS = {
    "maintenance": build_maintenance_agent,
    "cs": build_cs_agent,
    "vendor": build_vendor_agent,
    "finance": build_finance_agent,
}

AGENT_LABELS = {
    "maintenance": "Maintenance Agent",
    "cs": "Customer Service Agent",
    "vendor": "Vendor & Procurement Agent",
    "finance": "Finance Agent",
}

TOOL_LABELS = {
    "machine_failure_predictor": "model prediksi kegagalan mesin",
    "case_history_search": "pencarian kasus historis",
    "sop_retriever": "dokumen SOP resmi (RAG)",
    "complaint_category_classifier": "model klasifikasi kategori komplain (fine-tuned)",
    "vendor_document_validator": "checklist validasi dokumen vendor",
    "finance_weekly_summary": "ringkasan biaya operasional mingguan",
}

OUT_OF_SCOPE_MESSAGE = (
    "Maaf, pertanyaan ini berada di luar cakupan sistem PT Cipta Manufaktur Nusantara. "
    "Sistem ini hanya menangani pertanyaan seputar empat domain: (1) Maintenance -- "
    "kondisi mesin dan prediksi kegagalan, (2) Customer Service -- keluhan pelanggan "
    "dan SOP, (3) Vendor/Procurement -- validasi dokumen vendor, dan (4) Finance -- "
    "biaya operasional. Silakan ajukan pertanyaan yang berkaitan dengan salah satu "
    "domain tersebut, atau hubungi divisi terkait secara langsung untuk pertanyaan lain."
)


def _run_single_agent(agent_key: str, query: str, context: str = "") -> str:
    """Menjalankan satu agent secara terisolasi (Process.sequential, 1 agent, 1 task)."""
    agent = AGENT_BUILDERS[agent_key]()

    description = f'Jawab pertanyaan berikut dari staf/manajer: "{query}"'
    if context:
        description += (
            f"\n\nKonteks tambahan dari agent lain yang sudah dijalankan sebelumnya:\n{context}\n"
            "Gunakan konteks ini kalau relevan (misalnya untuk menghitung estimasi biaya "
            "berdasarkan data teknis yang sudah ditemukan)."
        )
    description += (
        "\n\nSelalu gunakan tools yang tersedia untuk mengambil data faktual sebelum "
        "menjawab -- jangan menjawab berdasarkan asumsi atau pengetahuan umum."
    )

    task = Task(
        description=description,
        expected_output="Jawaban faktual berdasarkan data yang diambil lewat tools, dalam Bahasa Indonesia.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()
    return str(result)


def _synthesize(query: str, agent_outputs: dict) -> str:
    """Menyatukan jawaban dari beberapa agent jadi satu jawaban akhir yang koheren.
    Kalau hanya 1 agent terlibat, tidak perlu sintesis -- langsung pakai jawabannya."""
    if len(agent_outputs) == 1:
        return list(agent_outputs.values())[0]

    context_blocks = "\n\n".join(
        f"[{AGENT_LABELS[k]}]:\n{v}" for k, v in agent_outputs.items()
    )
    synth_agent = AGENT_BUILDERS[list(agent_outputs.keys())[0]]()  # pakai LLM yang sama, role netral cukup
    task = Task(
        description=(
            f'Berikut adalah jawaban dari beberapa agent spesialis terhadap query: "{query}"\n\n'
            f"{context_blocks}\n\n"
            "Satukan jawaban-jawaban di atas menjadi satu jawaban akhir yang koheren dan "
            "runtut untuk staf/manajer. JANGAN menambahkan klaim atau data baru yang tidak "
            "ada pada jawaban di atas -- tugasmu murni merangkai, bukan menambah informasi."
        ),
        expected_output="Satu jawaban akhir yang koheren dalam Bahasa Indonesia, tanpa data tambahan yang tidak bersumber dari jawaban agent di atas.",
        agent=synth_agent,
    )
    crew = Crew(agents=[synth_agent], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()
    return str(result)


def _build_citation_section() -> str:
    """Membangun bagian sitasi sumber SECARA PROGRAMATIK dari tool_call_logger,
    bukan dari klaim LLM -- lihat penjelasan di docstring modul ini."""
    summary = tool_call_logger.summary_by_agent()
    if not summary:
        return "\n\n[Catatan: tidak ada tool yang tercatat terpanggil pada eksekusi ini.]"

    lines = ["\n\n---\nSumber yang benar-benar digunakan (tercatat otomatis dari log eksekusi):"]
    for role, tools in summary.items():
        tool_labels = [TOOL_LABELS.get(t, t) for t in tools]
        lines.append(f"- {role}: {', '.join(tool_labels)}")
    return "\n".join(lines)


def run_query(query: str) -> str:
    matched = classify_intent(query)

    if not matched:
        return OUT_OF_SCOPE_MESSAGE

    ordered = order_agents(matched)
    tool_call_logger.reset()

    agent_outputs = {}
    running_context = ""
    for agent_key in ordered:
        output = _run_single_agent(agent_key, query, context=running_context)
        agent_outputs[agent_key] = output
        running_context += f"\n[{AGENT_LABELS[agent_key]}]: {output}\n"

    final_answer = _synthesize(query, agent_outputs)
    final_answer += _build_citation_section()
    return final_answer


# --- Kompatibilitas dengan evaluator/kode lama yang memanggil build_crew() ---
def build_crew(query: str) -> Crew:
    """Dipertahankan untuk kompatibilitas (misal test perakitan crew di evaluator),
    TIDAK dipakai lagi oleh run_query(). Merakit satu Crew berisi semua agent
    dengan Process.sequential (bukan hierarchical) sebagai representasi statis."""
    agents = [builder() for builder in AGENT_BUILDERS.values()]
    task = Task(
        description=f'Tangani query berikut: "{query}"',
        expected_output="Jawaban faktual dalam Bahasa Indonesia.",
        agent=agents[0],
    )
    return Crew(agents=agents, tasks=[task], process=Process.sequential, verbose=True)
