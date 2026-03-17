from dataclasses import dataclass


@dataclass
class PromptResult:
    thread_id: str
    turn_id: str
    status: str
    final_text: str
