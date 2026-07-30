from crewai import Agent
from tools.failure_predictor_tool import FailurePredictorTool
from tools.case_search_tool import CaseSearchTool
from llm_config import get_llm


def build_maintenance_agent() -> Agent:
    return Agent(
        role="Maintenance Analyst",
        goal=(
            "Menganalisis kondisi mesin, memprediksi risiko kegagalan, dan memberikan "
            "rekomendasi tindakan preventif berdasarkan data sensor dan histori perbaikan."
        ),
        backstory=(
            "Kamu adalah analis maintenance senior di PT Cipta Manufaktur Nusantara dengan "
            "pengalaman 10 tahun menangani mesin CNC, conveyor, dan press. Kamu selalu "
            "mendasarkan rekomendasi pada data sensor aktual dan kasus historis serupa, "
            "bukan tebakan."
        ),
        tools=[FailurePredictorTool(), CaseSearchTool()],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
