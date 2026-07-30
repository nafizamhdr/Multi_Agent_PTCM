"""
tool_call_logger.py
Event listener CrewAI yang mencatat setiap tool yang BENAR-BENAR dipanggil
selama eksekusi (nama tool, agent yang memanggil, argumen, ringkasan output).

Kenapa ini penting?
Pada pengujian sebelumnya (Bab V.4.4 laporan), ditemukan bahwa jawaban akhir
orchestrator bisa mengklaim memakai sumber tertentu (SOP, dokumen vendor) padahal
tool terkait TIDAK PERNAH dipanggil -- LLM mengarang klaim sitasi. Untuk
menghilangkan risiko ini, sitasi sumber pada jawaban akhir sistem (lihat
crew.py) TIDAK LAGI diminta dari LLM, melainkan dibangun secara programatik
dari log tool call yang tercatat lewat listener ini -- ground truth eksekusi,
bukan klaim tekstual.
"""

from crewai.events import BaseEventListener
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent


class ToolCallLogger(BaseEventListener):
    def __init__(self):
        super().__init__()
        self.calls = []

    def reset(self):
        self.calls = []

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_used(source, event: ToolUsageFinishedEvent):
            self.calls.append({
                "agent_role": event.agent_role,
                "tool_name": event.tool_name,
                "tool_args": event.tool_args,
            })

    def summary_by_agent(self) -> dict:
        """Mengelompokkan tool yang terpanggil per agent, dipakai untuk membangun
        bagian 'Sumber yang digunakan' pada jawaban akhir secara akurat."""
        result = {}
        for call in self.calls:
            role = call["agent_role"] or "unknown"
            result.setdefault(role, [])
            if call["tool_name"] not in result[role]:
                result[role].append(call["tool_name"])
        return result


# Instance tunggal dipakai di seluruh proses (di-reset tiap awal run_query())
tool_call_logger = ToolCallLogger()
