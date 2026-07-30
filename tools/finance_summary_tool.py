"""
finance_summary_tool.py
CrewAI Tool untuk Finance Agent: ambil ringkasan biaya operasional dari laporan
agregasi mingguan (hasil gabungan data Maintenance & CS).
"""

import os
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "finance", "weekly_aggregated_report.csv"
)


class FinanceSummaryInput(BaseModel):
    n_weeks: int = Field(4, description="Jumlah minggu terakhir yang ingin dirangkum")


class FinanceSummaryTool(BaseTool):
    name: str = "finance_weekly_summary"
    description: str = (
        "Mengambil ringkasan biaya operasional N minggu terakhir, mencakup biaya maintenance, "
        "biaya kompensasi customer service, dan status vendor pending, beserta tren perubahannya. "
        "Data ini adalah hasil agregasi dari divisi Maintenance dan Customer Service."
    )
    args_schema: type[BaseModel] = FinanceSummaryInput

    def _run(self, n_weeks: int = 4) -> str:
        df = pd.read_csv(DATA_PATH)
        recent = df.tail(n_weeks)

        total_maint = recent["biaya_maintenance_idr"].sum()
        total_cs = recent["biaya_kompensasi_cs_idr"].sum()
        total_all = recent["total_biaya_operasional_idr"].sum()
        avg_weekly = recent["total_biaya_operasional_idr"].mean()
        vendor_pending = recent["vendor_pending_count"].iloc[-1] if len(recent) else "N/A"

        trend = "naik" if len(recent) >= 2 and recent["total_biaya_operasional_idr"].iloc[-1] > recent["total_biaya_operasional_idr"].iloc[0] else "turun/stabil"

        return (
            f"Ringkasan {n_weeks} minggu terakhir:\n"
            f"- Total biaya maintenance: Rp {total_maint:,.0f}\n"
            f"- Total biaya kompensasi CS: Rp {total_cs:,.0f}\n"
            f"- Total biaya operasional gabungan: Rp {total_all:,.0f}\n"
            f"- Rata-rata biaya per minggu: Rp {avg_weekly:,.0f}\n"
            f"- Jumlah vendor berstatus belum lengkap dokumen: {vendor_pending}\n"
            f"- Tren biaya operasional: {trend}"
        )
