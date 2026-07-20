"""
llm_config.py
Konfigurasi LLM backend untuk seluruh agent. Default: Ollama lokal.

Kalau USE_MOCK_LLM=true di .env (atau Ollama tidak terjangkau), semua agent
akan pakai MockLLM yang mengembalikan respons template statis -- ini dipakai
supaya skeleton tetap bisa didemokan alurnya (routing, tool-calling, evaluator)
walau Ollama belum di-setup di mesin yang menjalankan.
"""

import os
from dotenv import load_dotenv

load_dotenv()

USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def get_llm():
    """
    Mengembalikan objek LLM yang siap dipasang ke Agent CrewAI.
    CrewAI (via LiteLLM) support Ollama lewat prefix 'ollama/'.
    """
    if USE_MOCK_LLM:
        return None  # agent akan pakai MockLLM di level Task saat testing (lihat evaluator/mock_llm.py)

    from crewai import LLM
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


def check_ollama_available() -> bool:
    """Cek cepat apakah server Ollama bisa dihubungi. Berguna untuk main.py sebelum run."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False
