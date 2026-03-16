from __future__ import annotations

from Codex.Session import CodexSession
from Codex.Transport import AppServerTransport


class CodexClient:
    def __init__(self, executable: str = "codex"):
        self._transport = AppServerTransport(executable=executable)

    def start_session(self, cwd: str, dangerous: bool = False) -> CodexSession:
        thread_id = self._transport.start_thread(cwd=cwd, dangerous=dangerous)
        return CodexSession(
            transport=self._transport,
            thread_id=thread_id,
            cwd=cwd,
            dangerous=dangerous,
        )

    def resume_session(self, thread_id: str, cwd: str | None = None, dangerous: bool = False) -> CodexSession:
        resumed_thread_id = self._transport.resume_thread(
            thread_id=thread_id,
            cwd=cwd,
            dangerous=dangerous,
        )
        return CodexSession(
            transport=self._transport,
            thread_id=resumed_thread_id,
            cwd=cwd,
            dangerous=dangerous,
        )

    def close(self) -> None:
        self._transport.close()
