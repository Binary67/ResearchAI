from __future__ import annotations

from Codex.Logging import SessionLogger
from Codex.Session import CodexSession
from Codex.Transport import AppServerTransport


class CodexClient:
    def __init__(self, executable: str = "codex"):
        self._transport = AppServerTransport(executable=executable)
        self._session_loggers: list[SessionLogger] = []

    def start_session(self, cwd: str, dangerous: bool = False) -> CodexSession:
        logger = SessionLogger.create()
        try:
            thread_id = self._transport.start_thread(cwd=cwd, dangerous=dangerous)
        except Exception as exc:
            logger.write("Error", f"Failed to start session: {exc}")
            raise

        logger.write("Session", "Started Codex session.", body=f"cwd: {cwd}\ndangerous: {dangerous}")
        self._session_loggers.append(logger)
        return CodexSession(
            transport=self._transport,
            thread_id=thread_id,
            cwd=cwd,
            dangerous=dangerous,
            logger=logger,
        )

    def resume_session(self, thread_id: str, cwd: str | None = None, dangerous: bool = False) -> CodexSession:
        logger = SessionLogger.create()
        try:
            resumed_thread_id = self._transport.resume_thread(
                thread_id=thread_id,
                cwd=cwd,
                dangerous=dangerous,
            )
        except Exception as exc:
            logger.write("Error", f"Failed to resume session: {exc}")
            raise

        logger.write("Session", "Resumed Codex session.", body=f"cwd: {cwd or '(unchanged)'}\ndangerous: {dangerous}")
        self._session_loggers.append(logger)
        return CodexSession(
            transport=self._transport,
            thread_id=resumed_thread_id,
            cwd=cwd,
            dangerous=dangerous,
            logger=logger,
        )

    def close(self) -> None:
        for logger in self._session_loggers:
            logger.write("Session", "Closed Codex session.")
        self._session_loggers.clear()
        self._transport.close()
