from __future__ import annotations

from datetime import datetime
from pathlib import Path


_MAX_SUMMARY_LENGTH = 240
_MAX_BODY_LENGTH = 4000


class SessionLogger:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

    @classmethod
    def create(cls, logs_dir: str | Path = "Logs") -> "SessionLogger":
        directory = Path(logs_dir)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return cls(directory / f"codex-session-{timestamp}.log")

    @property
    def path(self) -> str:
        return str(self._path)

    def write(
        self,
        label: str,
        summary: str,
        body: str | None = None,
        body_limit: int | None = _MAX_BODY_LENGTH,
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_label = self._clean_line(label).strip() or "Log"
        clean_summary = self._truncate(self._clean_line(summary).strip(), _MAX_SUMMARY_LENGTH)

        lines = [f"{timestamp} [{clean_label}] {clean_summary}".rstrip()]

        clean_body = self._clean_body(body, body_limit=body_limit)
        if clean_body:
            for line in clean_body.splitlines():
                lines.append(f"  {line}" if line else "  ")

        with self._path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")

    def _clean_body(self, body: str | None, body_limit: int | None) -> str:
        if not body:
            return ""
        cleaned = body.replace("\r\n", "\n").replace("\r", "\n")
        if body_limit is not None:
            cleaned = self._truncate(cleaned, body_limit)
        return "\n".join(self._clean_line(line).rstrip() for line in cleaned.splitlines()).strip("\n")

    def _clean_line(self, value: str) -> str:
        return "".join(character if ord(character) < 128 else "?" for character in value)

    def _truncate(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."
