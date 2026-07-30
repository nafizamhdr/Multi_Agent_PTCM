from crewai import Agent
from tools.rag_tool import SOPRetrieverTool
from tools.case_search_tool import CaseSearchTool
from tools.complaint_classifier_tool import ComplaintClassifierTool
from llm_config import get_llm


def build_cs_agent() -> Agent:
    return Agent(
        role="Customer Service Specialist",
        goal=(
            "Menjawab dan menangani keluhan pelanggan secara akurat dan konsisten dengan "
            "SOP resmi perusahaan, serta merujuk pada kasus serupa yang pernah terjadi."
        ),
        backstory=(
            "Kamu adalah staf Customer Service berpengalaman yang selalu merujuk ke SOP "
            "resmi sebelum memberi keputusan ke pelanggan. Kamu tidak pernah menjanjikan "
            "sesuatu di luar SOP tanpa eskalasi ke supervisor. Kamu selalu mengklasifikasikan "
            "kategori komplain terlebih dahulu sebelum mencari SOP yang relevan."
        ),
        tools=[ComplaintClassifierTool(), SOPRetrieverTool(), CaseSearchTool()],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
