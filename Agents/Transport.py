from __future__ import annotations

import difflib
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

from Agents.Errors import CodexProcessError, CodexRpcError, CodexTurnError
from Agents.Logging import SessionLogger
from Agents.Types import PromptResult


_CLIENT_INFO = {
    "name": "researchai",
    "title": "ResearchAI",
    "version": "0.1.0",
}

_SNAPSHOT_IGNORE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "Logs",
    "node_modules",
}
_SNAPSHOT_IGNORE_FILES = {".DS_Store"}
_MAX_SNAPSHOT_FILE_BYTES = 200_000


@dataclass
class _TurnOutcome:
    prompt_result: PromptResult
    diff_text: str
    command_count: int
    file_changes: list[dict[str, Any]]


class AppServerTransport:
    def __init__(self, executable: str = "codex"):
        self._executable = executable
        self._process: subprocess.Popen[str] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=50)
        self._request_ids = count(1)
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._notifications: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._initialized = False

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            if not self._initialized:
                self.request("initialize", {"clientInfo": _CLIENT_INFO})
                self.notify("initialized", {})
                self._initialized = True
            return

        self._notifications = queue.Queue()
        self._pending = {}
        self._stderr_lines.clear()

        try:
            self._process = subprocess.Popen(
                [self._executable, "app-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise CodexProcessError(f"Failed to start `{self._executable} app-server`: {exc}") from exc

        if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
            self.close()
            raise CodexProcessError("Failed to create stdio pipes for `codex app-server`.")

        self._initialized = False
        self._reader_thread = threading.Thread(target=self._read_stdout, name="codex-stdout", daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, name="codex-stderr", daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()

        self.request("initialize", {"clientInfo": _CLIENT_INFO})
        self.notify("initialized", {})
        self._initialized = True

    def close(self) -> None:
        process = self._process
        self._process = None
        self._initialized = False

        if process is None:
            return

        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._process is None or self._process.poll() is not None:
            self.start()
        elif not self._initialized and method != "initialize":
            self.start()

        request_id = next(self._request_ids)
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_queue

        self._write_message({"method": method, "id": request_id, "params": params or {}})

        while True:
            try:
                response = response_queue.get(timeout=0.1)
                break
            except queue.Empty:
                self._raise_if_process_exited()

        if "error" in response:
            error = response["error"]
            raise CodexRpcError(error.get("code", -1), error.get("message", "Unknown JSON-RPC error"))

        return response.get("result", {})

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._process is None or self._process.poll() is not None:
            self.start()
        elif not self._initialized and method != "initialized":
            self.start()
        self._write_message({"method": method, "params": params or {}})

    def start_thread(self, cwd: str, dangerous: bool) -> str:
        params: dict[str, Any] = {"cwd": cwd}
        params.update(_dangerous_overrides(dangerous))
        result = self.request("thread/start", params)
        thread = result.get("thread") or {}
        thread_id = thread.get("id")
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexProcessError("`thread/start` did not return a thread id.")
        return thread_id

    def resume_thread(self, thread_id: str, cwd: str | None, dangerous: bool) -> str:
        params: dict[str, Any] = {"threadId": thread_id}
        if cwd is not None:
            params["cwd"] = cwd
        params.update(_dangerous_overrides(dangerous))
        result = self.request("thread/resume", params)
        thread = result.get("thread") or {}
        resumed_id = thread.get("id")
        if not isinstance(resumed_id, str) or not resumed_id:
            raise CodexProcessError("`thread/resume` did not return a thread id.")
        return resumed_id

    def prompt(
        self,
        thread_id: str,
        text: str,
        cwd: str | None,
        dangerous: bool,
        logger: SessionLogger | None = None,
    ) -> PromptResult:
        self._drain_notifications()

        if logger is not None:
            logger.write("Prompt", _prompt_summary(text), body=_prompt_body(cwd, dangerous))

        snapshot_before = _snapshot_workspace(cwd)
        turn_started_at = time.monotonic()

        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": text}],
        }
        if cwd is not None:
            params["cwd"] = cwd
        params.update(_dangerous_overrides(dangerous))

        try:
            result = self.request("turn/start", params)
            turn = result.get("turn") or {}
            turn_id = turn.get("id")
            if not isinstance(turn_id, str) or not turn_id:
                raise CodexProcessError("`turn/start` did not return a turn id.")

            outcome = self._collect_turn_result(thread_id=thread_id, turn_id=turn_id, logger=logger)
        except CodexTurnError:
            snapshot_after = _snapshot_workspace(cwd)
            file_count = _log_file_changes(
                logger=logger,
                cwd=cwd,
                diff_text="",
                file_changes=[],
                snapshot_before=snapshot_before,
                snapshot_after=snapshot_after,
            )
            if logger is not None:
                logger.write(
                    "Turn",
                    f"Failed after {time.monotonic() - turn_started_at:.2f}s.",
                    body=f"files changed: {file_count}" if file_count else None,
                )
            raise
        except (CodexProcessError, CodexRpcError) as exc:
            snapshot_after = _snapshot_workspace(cwd)
            file_count = _log_file_changes(
                logger=logger,
                cwd=cwd,
                diff_text="",
                file_changes=[],
                snapshot_before=snapshot_before,
                snapshot_after=snapshot_after,
            )
            if logger is not None:
                logger.write("Error", f"Turn failed: {exc}", body=self._stderr_summary())
                logger.write(
                    "Turn",
                    f"Failed after {time.monotonic() - turn_started_at:.2f}s.",
                    body=f"files changed: {file_count}" if file_count else None,
                )
            raise

        snapshot_after = _snapshot_workspace(cwd)
        file_count = _log_file_changes(
            logger=logger,
            cwd=cwd,
            diff_text=outcome.diff_text,
            file_changes=outcome.file_changes,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

        if logger is not None:
            logger.write(
                "Answer",
                "Final response.",
                outcome.prompt_result.final_text or "(empty response)",
                body_limit=None,
            )
            logger.write(
                "Turn",
                f"Completed in {time.monotonic() - turn_started_at:.2f}s.",
                body=(
                    f"status: {outcome.prompt_result.status}\n"
                    f"commands: {outcome.command_count}\n"
                    f"files changed: {file_count}"
                ),
            )

        return outcome.prompt_result

    def _collect_turn_result(
        self,
        thread_id: str,
        turn_id: str,
        logger: SessionLogger | None = None,
    ) -> _TurnOutcome:
        agent_messages: list[dict[str, Any]] = []
        delta_text_by_item: dict[str, str] = {}
        diff_text = ""
        command_count = 0
        file_changes: list[dict[str, Any]] = []
        generic_thinking_logged = False

        while True:
            try:
                notification = self._notifications.get(timeout=0.1)
            except queue.Empty:
                self._raise_if_process_exited()
                continue

            method = notification.get("method")
            params = notification.get("params") or {}

            if method == "item/completed":
                item = params.get("item") or {}
                item_type = item.get("type")

                if item_type == "reasoning":
                    summary = _reasoning_summary(item)
                    if logger is not None and summary:
                        logger.write("Thinking", summary)
                        generic_thinking_logged = True
                    continue

                if item_type == "agentMessage":
                    agent_messages.append(item)
                    commentary = _commentary_summary(item)
                    if logger is not None and commentary:
                        logger.write("Thinking", commentary)
                        generic_thinking_logged = True
                    continue

                if item_type == "fileChange":
                    changes = item.get("changes")
                    if isinstance(changes, list):
                        file_changes.extend(change for change in changes if isinstance(change, dict))
                    continue

                if _is_command_item(item):
                    command_count += 1
                    if logger is not None:
                        _log_command_item(logger, item)
                    continue

                continue

            if method == "item/agentMessage/delta":
                item_id = params.get("itemId") or params.get("id")
                delta_text = _extract_delta_text(params)
                if isinstance(item_id, str) and delta_text:
                    delta_text_by_item[item_id] = delta_text_by_item.get(item_id, "") + delta_text
                if logger is not None and delta_text.strip() and not generic_thinking_logged:
                    logger.write("Thinking", "Codex is working on the request.")
                    generic_thinking_logged = True
                continue

            if method == "turn/diff/updated":
                candidate_diff = params.get("diff")
                if isinstance(candidate_diff, str) and candidate_diff.strip():
                    diff_text = candidate_diff
                continue

            if method != "turn/completed":
                continue

            completed_turn = params.get("turn") or {}
            if completed_turn.get("id") != turn_id:
                continue

            status = completed_turn.get("status", "unknown")
            if status != "completed":
                error = completed_turn.get("error") or {}
                message = error.get("message", f"Turn finished with status `{status}`.")
                if logger is not None:
                    logger.write("Error", f"Turn failed with status {status}.", body=message)
                raise CodexTurnError(status=status, message=message, thread_id=thread_id, turn_id=turn_id)

            return _TurnOutcome(
                prompt_result=PromptResult(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    status=status,
                    final_text=_choose_final_text(agent_messages, delta_text_by_item),
                ),
                diff_text=diff_text,
                command_count=command_count,
                file_changes=file_changes,
            )

    def _drain_notifications(self) -> None:
        while True:
            try:
                self._notifications.get_nowait()
            except queue.Empty:
                return

    def _write_message(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise CodexProcessError("`codex app-server` is not running.")

        try:
            payload = json.dumps(message, separators=(",", ":"))
            with self._write_lock:
                process.stdin.write(f"{payload}\n")
                process.stdin.flush()
        except OSError as exc:
            raise CodexProcessError(f"Failed to write to `codex app-server`: {exc}") from exc

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        for line in process.stdout:
            message_text = line.strip()
            if not message_text:
                continue

            try:
                message = json.loads(message_text)
            except json.JSONDecodeError:
                continue

            if "id" in message:
                request_id = message["id"]
                with self._pending_lock:
                    response_queue = self._pending.pop(request_id, None)
                if response_queue is not None:
                    response_queue.put(message)
                continue

            self._notifications.put(message)

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return

        for line in process.stderr:
            message_text = line.rstrip()
            if message_text:
                self._stderr_lines.append(message_text)

    def _raise_if_process_exited(self) -> None:
        process = self._process
        if process is None:
            raise CodexProcessError("`codex app-server` is not running.")

        return_code = process.poll()
        if return_code is None:
            return

        stderr_output = "\n".join(self._stderr_lines)
        detail = f" Exit code: {return_code}."
        if stderr_output:
            detail = f"{detail} stderr:\n{stderr_output}"
        raise CodexProcessError(f"`codex app-server` exited unexpectedly.{detail}")

    def _stderr_summary(self) -> str | None:
        if not self._stderr_lines:
            return None
        return "\n".join(self._stderr_lines)


def _dangerous_overrides(dangerous: bool) -> dict[str, Any]:
    if not dangerous:
        return {}
    return {
        "approvalPolicy": "never",
        "sandboxPolicy": {"type": "dangerFullAccess"},
    }


def _extract_delta_text(params: dict[str, Any]) -> str:
    for key in ("delta", "textDelta", "text"):
        value = params.get(key)
        if isinstance(value, str):
            return value
    return ""


def _choose_final_text(agent_messages: list[dict[str, Any]], delta_text_by_item: dict[str, str]) -> str:
    final_messages = [item for item in agent_messages if item.get("phase") == "final_answer"]
    if final_messages:
        text = final_messages[-1].get("text")
        if isinstance(text, str):
            return text

    if agent_messages:
        text = agent_messages[-1].get("text")
        if isinstance(text, str):
            return text

    if delta_text_by_item:
        return next(reversed(delta_text_by_item.values()))

    return ""


def _prompt_summary(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "Empty prompt."
    return stripped.splitlines()[0]


def _prompt_body(cwd: str | None, dangerous: bool) -> str:
    return f"cwd: {cwd or '(default)'}\ndangerous: {dangerous}"


def _reasoning_summary(item: dict[str, Any]) -> str:
    summary = _extract_text(item.get("summary"))
    if summary:
        return summary
    return _extract_text(item.get("content"))


def _commentary_summary(item: dict[str, Any]) -> str:
    if item.get("phase") != "commentary":
        return ""
    text = item.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def _is_command_item(item: dict[str, Any]) -> bool:
    item_type = str(item.get("type", "")).lower()
    return "command" in item_type or "exec" in item_type or item_type.startswith("tool")


def _log_command_item(logger: SessionLogger, item: dict[str, Any]) -> None:
    summary = _command_summary(item)
    body_lines: list[str] = []

    status = item.get("status")
    if isinstance(status, str) and status:
        body_lines.append(f"status: {status}")

    exit_code = _extract_int(item, "exitCode", "exit_code", "code")
    if exit_code is not None:
        body_lines.append(f"exit code: {exit_code}")

    duration_ms = _extract_int(item, "durationMs", "duration_ms")
    if duration_ms is not None:
        body_lines.append(f"duration: {duration_ms} ms")

    logger.write("Command", summary, body="\n".join(body_lines) or None)

    output = _command_output(item)
    if not output:
        return

    failed = (exit_code is not None and exit_code != 0) or (
        isinstance(status, str) and status not in {"completed", "success"}
    )
    logger.write(
        "Command Output",
        "Command failed." if failed else "Command output.",
        output,
        body_limit=None if failed else 1000,
    )


def _command_summary(item: dict[str, Any]) -> str:
    for key in ("command", "cmd", "shellCommand", "input"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    tool_name = item.get("name") or item.get("toolName")
    arguments = item.get("arguments") or item.get("args")
    if isinstance(tool_name, str) and tool_name.strip():
        argument_text = _extract_text(arguments)
        if argument_text:
            return f"{tool_name.strip()} {argument_text}"
        return tool_name.strip()

    item_type = str(item.get("type", "command")) or "command"
    return f"Completed {item_type}."


def _command_output(item: dict[str, Any]) -> str:
    for key in ("output", "stdout", "stderr", "result"):
        value = item.get(key)
        text = _extract_text(value)
        if text:
            return text
    return ""


def _extract_int(item: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, int):
            return value
    return None


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = [_extract_text(part) for part in value]
        return "\n".join(part for part in parts if part).strip()

    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            return value["text"].strip()

        preferred_keys = ("text", "summary", "content", "message", "msg", "value")
        parts = [_extract_text(value.get(key)) for key in preferred_keys if key in value]
        if any(parts):
            return "\n".join(part for part in parts if part).strip()

        nested_parts = [_extract_text(part) for part in value.values()]
        return "\n".join(part for part in nested_parts if part).strip()

    return ""


def _log_file_changes(
    logger: SessionLogger | None,
    cwd: str | None,
    diff_text: str,
    file_changes: list[dict[str, Any]],
    snapshot_before: dict[str, str] | None,
    snapshot_after: dict[str, str] | None,
) -> int:
    if logger is None:
        return 0

    entries = _entries_from_turn_diff(diff_text, cwd)
    if not entries:
        entries = _entries_from_file_changes(file_changes, cwd)
    if not entries:
        entries = _entries_from_snapshots(snapshot_before, snapshot_after)

    for action, path, body in entries:
        logger.write("File Changed", f"{action} {path}", body=body or None)

    return len(entries)


def _entries_from_turn_diff(diff_text: str, cwd: str | None) -> list[tuple[str, str, str]]:
    if not diff_text.strip():
        return []

    entries: list[tuple[str, str, str]] = []
    current_lines: list[str] = []

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git ") and current_lines:
            entries.append(_diff_entry_from_block("".join(current_lines), cwd))
            current_lines = [line]
            continue
        current_lines.append(line)

    if current_lines:
        entries.append(_diff_entry_from_block("".join(current_lines), cwd))

    return [entry for entry in entries if entry is not None]


def _diff_entry_from_block(block: str, cwd: str | None) -> tuple[str, str, str] | None:
    path = ""
    for line in block.splitlines():
        if line.startswith("+++ "):
            candidate = _normalize_diff_path(line[4:].strip(), cwd)
            if candidate:
                path = candidate
                break

    if not path:
        for line in block.splitlines():
            if line.startswith("--- "):
                candidate = _normalize_diff_path(line[4:].strip(), cwd)
                if candidate:
                    path = candidate
                    break

    if not path:
        return None

    action = "Updated"
    if "new file mode" in block:
        action = "Created"
    elif "deleted file mode" in block:
        action = "Deleted"

    return action, path, _rewrite_diff_block(block, path)


def _normalize_diff_path(raw_path: str, cwd: str | None) -> str:
    if raw_path == "/dev/null":
        return ""

    normalized = raw_path
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]

    candidate = Path(normalized)
    if candidate.is_absolute() and cwd is not None:
        try:
            return candidate.resolve(strict=False).relative_to(Path(cwd).resolve(strict=False)).as_posix()
        except ValueError:
            return candidate.resolve(strict=False).as_posix()

    return candidate.as_posix()


def _entries_from_file_changes(
    file_changes: list[dict[str, Any]],
    cwd: str | None,
) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for change in file_changes:
        raw_path = change.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue

        display_path = _display_path(raw_path, cwd)
        action = _change_action(change)
        diff_body = change.get("diff")
        body = diff_body if isinstance(diff_body, str) and diff_body.strip() else ""
        entries.append((action, display_path, body.rstrip()))
    return entries


def _change_action(change: dict[str, Any]) -> str:
    kind = change.get("kind")
    if isinstance(kind, dict):
        kind_type = kind.get("type")
        if isinstance(kind_type, str):
            if kind_type == "add":
                return "Created"
            if kind_type == "delete":
                return "Deleted"
    return "Updated"


def _display_path(raw_path: str, cwd: str | None) -> str:
    path = Path(raw_path)
    if path.is_absolute() and cwd is not None:
        try:
            return path.resolve(strict=False).relative_to(Path(cwd).resolve(strict=False)).as_posix()
        except ValueError:
            return path.resolve(strict=False).as_posix()
    return path.as_posix()


def _rewrite_diff_block(block: str, path: str) -> str:
    rewritten_lines: list[str] = []
    for line in block.splitlines():
        if line.startswith("diff --git "):
            rewritten_lines.append(f"diff --git a/{path} b/{path}")
            continue
        if line.startswith("--- ") and "/dev/null" not in line:
            rewritten_lines.append(f"--- a/{path}")
            continue
        if line.startswith("+++ ") and "/dev/null" not in line:
            rewritten_lines.append(f"+++ b/{path}")
            continue
        rewritten_lines.append(line)
    return "\n".join(rewritten_lines).rstrip()


def _entries_from_snapshots(
    snapshot_before: dict[str, str] | None,
    snapshot_after: dict[str, str] | None,
) -> list[tuple[str, str, str]]:
    if snapshot_before is None or snapshot_after is None:
        return []

    entries: list[tuple[str, str, str]] = []
    all_paths = sorted(set(snapshot_before) | set(snapshot_after))
    for path in all_paths:
        before_text = snapshot_before.get(path)
        after_text = snapshot_after.get(path)
        if before_text == after_text:
            continue

        action = "Updated"
        if before_text is None:
            action = "Created"
        elif after_text is None:
            action = "Deleted"

        diff_body = "".join(
            difflib.unified_diff(
                [] if before_text is None else before_text.splitlines(keepends=True),
                [] if after_text is None else after_text.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            )
        ).rstrip()
        entries.append((action, path, diff_body))
    return entries


def _snapshot_workspace(cwd: str | None) -> dict[str, str] | None:
    if cwd is None:
        return None

    root = Path(cwd)
    if not root.is_dir():
        return None

    snapshot: dict[str, str] = {}
    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(name for name in dir_names if name not in _SNAPSHOT_IGNORE_DIRS)
        current_path = Path(current_root)

        for file_name in sorted(file_names):
            if file_name in _SNAPSHOT_IGNORE_FILES:
                continue

            file_path = current_path / file_name
            relative_path = file_path.relative_to(root).as_posix()

            try:
                if not file_path.is_file():
                    continue
                if file_path.stat().st_size > _MAX_SNAPSHOT_FILE_BYTES:
                    continue
                snapshot[relative_path] = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

    return snapshot
