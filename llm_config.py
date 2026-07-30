"""
llm_config.py
Konfigurasi LLM backend untuk seluruh agent. Sistem ini menggunakan Ollama
(lokal, gratis, tanpa rate limit) sebagai satu-satunya provider.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def get_llm():
    """Mengembalikan objek LLM Ollama yang siap dipasang ke Agent CrewAI."""
    from crewai import LLM
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


def check_llm_available() -> bool:
    """Cek cepat apakah server Ollama bisa dihubungi. Dipakai main.py sebelum run."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False
