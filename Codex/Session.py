from __future__ import annotations

from Codex.Logging import SessionLogger
from Codex.Transport import AppServerTransport
from Codex.Types import PromptResult


class CodexSession:
    def __init__(
        self,
        transport: AppServerTransport,
        thread_id: str,
        cwd: str | None,
        dangerous: bool,
        logger: SessionLogger,
    ):
        self._transport = transport
        self._thread_id = thread_id
        self._cwd = cwd
        self._dangerous = dangerous
        self._logger = logger

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @property
    def log_path(self) -> str:
        return self._logger.path

    def prompt(self, text: str) -> PromptResult:
        return self._transport.prompt(
            thread_id=self._thread_id,
            text=text,
            cwd=self._cwd,
            dangerous=self._dangerous,
            logger=self._logger,
        )
