"""
rag_tool.py
CrewAI Tool untuk retrieve potongan dokumen SOP yang relevan dengan query CS.
"""

import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tools.vector_store import VectorStore

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


class SOPRetrieverInput(BaseModel):
    query: str = Field(..., description="Pertanyaan atau isi komplain pelanggan yang perlu dicari SOP-nya")
    n_results: int = Field(3, description="Jumlah potongan SOP yang diambil")


class SOPRetrieverTool(BaseTool):
    name: str = "sop_retriever"
    description: str = (
        "Mencari potongan dokumen SOP (Standard Operating Procedure) Customer Service "
        "yang paling relevan dengan sebuah query/komplain pelanggan. "
        "Gunakan tool ini untuk mendapatkan dasar prosedur resmi sebelum menjawab komplain."
    )
    args_schema: type[BaseModel] = SOPRetrieverInput

    def _run(self, query: str, n_results: int = 3) -> str:
        store = VectorStore(persist_dir=CHROMA_DIR)
        results = store.query("sop_documents", query, n_results=n_results)

        if not results["documents"][0]:
            return "Tidak ditemukan SOP yang relevan untuk query ini."

        output = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            output.append(
                f"[Sumber: {meta.get('source_file')}, skor_jarak={dist:.3f}]\n{doc}"
            )
        return "\n\n---\n\n".join(output)
