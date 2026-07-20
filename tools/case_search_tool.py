"""
case_search_tool.py
CrewAI Tool untuk mencari kasus historis yang mirip (maintenance atau komplain CS)
berdasarkan semantic/TF-IDF similarity. Dipakai Maintenance Agent & CS Agent.
"""

import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tools.vector_store import VectorStore

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


class CaseSearchInput(BaseModel):
    query: str = Field(..., description="Deskripsi kasus/masalah yang ingin dicari kemiripannya")
    n_results: int = Field(3, description="Jumlah kasus historis yang diambil")


class CaseSearchTool(BaseTool):
    name: str = "case_history_search"
    description: str = (
        "Mencari kasus historis (maintenance mesin ATAU komplain pelanggan) yang paling "
        "mirip dengan deskripsi masalah saat ini. Berguna untuk melihat bagaimana kasus "
        "serupa ditangani sebelumnya, termasuk downtime, biaya, dan resolusi."
    )
    args_schema: type[BaseModel] = CaseSearchInput

    def _run(self, query: str, n_results: int = 3) -> str:
        store = VectorStore(persist_dir=CHROMA_DIR)
        results = store.query("case_history", query, n_results=n_results)

        if not results["documents"][0]:
            return "Tidak ditemukan kasus historis yang relevan."

        output = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if k != "type")
            output.append(f"[{meta.get('type')}, skor_jarak={dist:.3f}, {meta_str}]\n{doc}")
        return "\n\n---\n\n".join(output)
