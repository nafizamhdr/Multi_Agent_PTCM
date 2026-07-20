"""
crew.py
Merakit Crew utama: Orchestrator (manager, otomatis dari CrewAI Process.hierarchical)
mendelegasikan task ke 4 agent spesialis (Maintenance, CS, Vendor, Finance), lalu
menyatukan hasilnya jadi satu jawaban akhir.

Kenapa hierarchical process?
Supaya "interaksi antar-agent" nyata: manager membaca query, memutuskan agent mana
yang relevan (bisa lebih dari satu), delegasikan task, lalu mensintesis jawaban --
bukan sekadar memanggil 4 agent secara terpisah lalu digabung manual.
"""

from crewai import Crew, Task, Process

from agents.maintenance_agent import build_maintenance_agent
from agents.cs_agent import build_cs_agent
from agents.vendor_agent import build_vendor_agent
from agents.finance_agent import build_finance_agent
from llm_config import get_llm


def build_crew(query: str) -> Crew:
    maintenance_agent = build_maintenance_agent()
    cs_agent = build_cs_agent()
    vendor_agent = build_vendor_agent()
    finance_agent = build_finance_agent()

    # Task tunggal yang deskripsinya berisi query user.
    # Di hierarchical process, manager (orchestrator) yang memutuskan agent mana
    # yang mengerjakan task ini -- termasuk kalau perlu delegasi ke lebih dari satu agent
    # secara berurutan (misal Maintenance Agent dulu, hasilnya dipakai Finance Agent).
    main_task = Task(
        description=(
            f"Tangani query berikut dari staf/manajer PT Cipta Manufaktur Nusantara: "
            f"\"{query}\"\n\n"
            "Instruksi:\n"
            "1. Identifikasi divisi mana yang relevan (Maintenance, Customer Service, "
            "Vendor/HR, dan/atau Finance).\n"
            "2. Delegasikan ke agent spesialis yang sesuai. Kalau query membutuhkan "
            "informasi dari lebih dari satu divisi (misal estimasi biaya akibat downtime "
            "mesin), delegasikan berurutan: dapatkan data teknis dulu, baru delegasikan "
            "ke Finance Agent untuk hitung estimasi biayanya.\n"
            "3. Sintesiskan semua hasil delegasi menjadi satu jawaban akhir yang koheren, "
            "sertakan sumber/dasar (SOP, data sensor, histori kasus, dokumen vendor) yang "
            "dipakai tiap agent."
        ),
        expected_output=(
            "Jawaban akhir yang koheren dalam Bahasa Indonesia, terstruktur dengan jelas, "
            "menyebutkan agent/divisi mana saja yang berkontribusi dan dasar informasi yang "
            "dipakai (nama SOP, hasil prediksi model, atau data vendor/finance yang relevan)."
        ),
        agent=None,  # dibiarkan kosong -- manager yang assign di hierarchical process
    )

    crew = Crew(
        agents=[maintenance_agent, cs_agent, vendor_agent, finance_agent],
        tasks=[main_task],
        process=Process.hierarchical,
        manager_llm=get_llm(),
        verbose=True,
    )
    return crew


def run_query(query: str) -> str:
    crew = build_crew(query)
    result = crew.kickoff()
    return str(result)
