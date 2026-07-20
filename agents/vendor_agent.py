from crewai import Agent
from tools.vendor_validator_tool import VendorValidatorTool
from llm_config import get_llm


def build_vendor_agent() -> Agent:
    return Agent(
        role="Vendor & Procurement Analyst",
        goal=(
            "Memvalidasi kelengkapan dan kesesuaian dokumen vendor terhadap checklist "
            "standar perusahaan, serta mendeteksi risiko yang perlu diklarifikasi sebelum "
            "onboarding disetujui."
        ),
        backstory=(
            "Kamu adalah staf HR/Procurement yang bertanggung jawab memastikan setiap "
            "vendor baru memenuhi standar dokumen sebelum bisa mulai bekerja sama, supaya "
            "tidak ada masalah legal atau kualitas di kemudian hari."
        ),
        tools=[VendorValidatorTool()],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
