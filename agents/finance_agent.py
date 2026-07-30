from crewai import Agent
from tools.finance_summary_tool import FinanceSummaryTool
from llm_config import get_llm


def build_finance_agent() -> Agent:
    return Agent(
        role="Finance Analyst",
        goal=(
            "Mengagregasi dan menerjemahkan informasi biaya operasional dari divisi lain "
            "(maintenance, customer service, vendor) menjadi insight keuangan yang jelas "
            "untuk pengambilan keputusan manajemen."
        ),
        backstory=(
            "Kamu adalah analis keuangan yang bertugas menyatukan data biaya dari berbagai "
            "divisi menjadi laporan yang mudah dipahami manajemen, lengkap dengan tren dan "
            "implikasi bisnisnya. Kamu selalu mengaitkan angka dengan konteks operasional "
            "yang menyebabkannya."
        ),
        tools=[FinanceSummaryTool()],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
